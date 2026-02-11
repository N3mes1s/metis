# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import unidiff
import pathspec

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from concurrent.futures import ThreadPoolExecutor, as_completed
from metis.configuration import load_plugin_config
from metis.exceptions import (
    PluginNotFoundError,
    QueryEngineInitError,
    ParsingError,
)
from metis.vector_store.base import BaseVectorStore
from metis.plugin_loader import load_plugins, discover_supported_language_names
from metis.utils import (
    count_tokens,
    read_file_content,
)

from .helpers import (
    build_summary_chain,
    summarize_changes,
    prepare_nodes_iter,
    apply_custom_guidance,
)
from .diff_utils import extract_content_from_diff, process_diff_file
from .graphs.types import ReviewRequest
from .graphs.types import AskRequest
from metis.engine.graphs import ReviewGraph, AskGraph


logger = logging.getLogger("metis")

# Sentinel used to distinguish "cache populated with None" from "not yet cached".
_METISIGNORE_MISS = object()


class MetisEngine:

    _SUPPORTED_LANGUAGES = None

    def __init__(
        self,
        codebase_path=".",
        vector_backend=BaseVectorStore,
        llm_provider=None,
        **kwargs,
    ):
        self.codebase_path = codebase_path
        self.vector_backend = vector_backend

        required_keys = [
            "max_workers",
            "max_token_length",
            "llama_query_model",
            "similarity_top_k",
            "response_mode",
        ]
        missing = [k for k in required_keys if k not in kwargs or kwargs[k] is None]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")

        for k in required_keys:
            setattr(self, k, kwargs[k])

        self.llm_provider = llm_provider
        self.doc_chunk_size = kwargs.get("doc_chunk_size", 1024)
        self.doc_chunk_overlap = kwargs.get("doc_chunk_overlap", 200)
        # Optional user-provided guidance to be appended to system prompts
        self.custom_prompt_text = kwargs.get("custom_prompt_text")
        self.plugin_config = load_plugin_config()

        # Load precedence note from general prompts
        self.custom_guidance_precedence = self.plugin_config.get(
            "general_prompts", {}
        ).get("custom_guidance_precedence", "")
        self.plugins = load_plugins(self.plugin_config)

        # Cache splitters and extension/plugin lookups
        self._splitter_cache = {}
        self.code_exts = set()
        self.ext_plugin_map = {}

        for plugin in self.plugins:
            for e in plugin.get_supported_extensions():
                e_lower = e.lower()
                self.code_exts.add(e_lower)
                self.ext_plugin_map[e_lower] = plugin

        # Graphs are built lazily on first use
        self._review_graph = None
        self._ask_graph = None
        self.metisignore_file = kwargs.get("metisignore_file") or ".metisignore"
        self.review_code_include_paths = kwargs.get("review_code_include_paths", [])
        self.review_code_exclude_paths = kwargs.get("review_code_exclude_paths", [])

    def load_metisignore(self) -> pathspec.GitIgnoreSpec | None:
        """
        Load metisignore file and return a GitIgnoreSpec matcher.
        The result is cached for the lifetime of this engine instance.

        Returns:
            pathspec.GitIgnoreSpec object or None if file doesn't exist
        """
        cached = getattr(self, "_metisignore_cache", None)
        if cached is not None:
            return cached if cached is not _METISIGNORE_MISS else None

        try:
            if not self.metisignore_file:
                logger.info("No MetisIgnore file provided")
                self._metisignore_cache = _METISIGNORE_MISS
                return None
            with open(self.metisignore_file, "r") as f:
                spec = pathspec.GitIgnoreSpec.from_lines(f)
                logger.info(f"MetisIgnore file loaded: {self.metisignore_file}")
            self._metisignore_cache = spec
            return spec
        except FileNotFoundError:
            logger.info(f"MetisIgnore file not loaded {self.metisignore_file}")
            self._metisignore_cache = _METISIGNORE_MISS
            return None

    def _get_review_graph(self):
        if self._review_graph is None:
            self._review_graph = ReviewGraph(
                llm_provider=self.llm_provider,
                plugin_config=self.plugin_config,
                custom_prompt_text=self.custom_prompt_text,
                custom_guidance_precedence=self.custom_guidance_precedence,
                llama_query_model=self.llama_query_model,
                max_token_length=self.max_token_length,
            )
        return self._review_graph

    def _get_ask_graph(self):
        if self._ask_graph is None:
            self._ask_graph = AskGraph(
                llm_provider=self.llm_provider,
                llama_query_model=self.llama_query_model,
            )
        return self._ask_graph

    @classmethod
    def supported_languages(cls):
        """
        Returns the list of supported languages by the Metis engine.
        """
        # Cache to avoid repeated plugin instantiation in repeated calls
        if cls._SUPPORTED_LANGUAGES is None:
            plugin_config = load_plugin_config()
            cls._SUPPORTED_LANGUAGES = discover_supported_language_names(plugin_config)
        return cls._SUPPORTED_LANGUAGES

    def get_plugin_from_name(self, name):
        for plugin in self.plugins:
            if (
                hasattr(plugin, "get_name")
                and plugin.get_name().lower() == name.lower()
            ):
                return plugin
        logger.error(f"Plugin '{name}' not found.")
        raise PluginNotFoundError(name)

    def _get_plugin_for_extension(self, extension):
        return self.ext_plugin_map.get(extension.lower())

    def _get_all_supported_code_extensions(self):
        return sorted(self.code_exts)

    def _get_splitter_cached(self, plugin):
        key = plugin.get_name()
        if key in self._splitter_cache:
            return self._splitter_cache[key]
        splitter = plugin.get_splitter()
        self._splitter_cache[key] = splitter
        return splitter

    def _get_doc_splitter(self):
        if not hasattr(self, "_doc_splitter") or self._doc_splitter is None:
            self._doc_splitter = SentenceSplitter(
                chunk_size=self.doc_chunk_size,
                chunk_overlap=self.doc_chunk_overlap,
            )
        return self._doc_splitter

    def _rel_to_base(self, path):
        base_path = os.path.abspath(self.codebase_path)
        return base_path, os.path.relpath(path, base_path)

    def ask_question(self, question):
        """
        Loads the indexes and queries them for an answer using the AskGraph.
        """
        qe_code, qe_docs = self._init_and_get_query_engines()
        logger.info("Querying codebase for your question...")
        req: AskRequest = {
            "question": question,
            "retriever_code": qe_code,
            "retriever_docs": qe_docs,
        }
        return self._get_ask_graph().ask(req)

    def index_codebase(self):
        """
        Reads files from the codebase, splits documents using language-specific
        splitters, builds vector indexes for code and documentation, and persists them.
        """

        self.index_prepare_nodes()
        self.index_finalize_embeddings()

    def index_prepare_nodes_iter(self):
        """
        Parse documents and prepare nodes for indexing, yielding one step per file.
        Stores prepared nodes internally for a subsequent call to
        `index_finalize_embeddings`.
        """
        # Read docs and code supported extensions from config
        docs_supported_exts = self.plugin_config.get("docs", {}).get(
            "supported_extensions", [".md"]
        )
        code_supported_exts = self._get_all_supported_code_extensions()

        logger.info(f"Indexing codebase at: {self.codebase_path}")
        reader = SimpleDirectoryReader(
            input_dir=self.codebase_path,
            recursive=True,
            required_exts=code_supported_exts + docs_supported_exts,
            filename_as_id=True,
        )
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from {self.codebase_path}")

        self.vector_backend.init()
        doc_splitter = self._get_doc_splitter()
        metisignore_spec = self.load_metisignore()
        base_path = os.path.abspath(self.codebase_path)
        parent_dir = os.path.dirname(base_path)
        code_docs = []
        doc_docs = []
        for doc in documents:
            ext = os.path.splitext(doc.id_)[1].lower()
            new_id = os.path.relpath(doc.id_, parent_dir)
            doc.doc_id = new_id
            doc.id_ = new_id

            if metisignore_spec and metisignore_spec.match_file(
                os.path.join(parent_dir, new_id)
            ):
                continue

            if ext in docs_supported_exts:
                doc_docs.append(doc)
            elif ext in code_supported_exts:
                code_docs.append(doc)

        nodes_code, nodes_docs = yield from prepare_nodes_iter(
            code_docs,
            doc_docs,
            self._get_plugin_for_extension,
            self._get_splitter_cached,
            doc_splitter,
        )

        # Store nodes for embedding phase
        self._pending_nodes = (nodes_code, nodes_docs)
        return

    def index_prepare_nodes(self):
        """
        Prepare nodes without exposing an iterator.
        Consumes the iterator so non-verbose callers avoid a no-op loop.
        """
        for _ in self.index_prepare_nodes_iter():
            pass

    def index_finalize_embeddings(self):
        """Build vector indexes from previously prepared nodes."""
        pending = getattr(self, "_pending_nodes", None)
        if not pending:
            # Nothing to do
            return
        nodes_code, nodes_docs = pending
        storage_context_code, storage_context_docs = (
            self.vector_backend.get_storage_contexts()
        )
        VectorStoreIndex(
            nodes_code,
            storage_context=storage_context_code,
            embed_model=self.llm_provider.get_embed_model_code(),
        )

        VectorStoreIndex(
            nodes_docs,
            storage_context=storage_context_docs,
            embed_model=self.llm_provider.get_embed_model_docs(),
        )
        # Clear pending nodes
        self._pending_nodes = None

    def review_file(self, file_path, skip_retrieval=False):
        """
        Review a single source file. Detects plugin by extension, retrieves
        relevant context from code/docs indexes, runs the security review,
        and returns a result dict or None
        if the file is unsupported or empty.
        """
        if skip_retrieval:
            qe_code, qe_docs = None, None
        else:
            qe_code, qe_docs = self._init_and_get_query_engines()
        base_path = os.path.abspath(self.codebase_path)
        snippet = read_file_content(file_path)
        if not snippet:
            return None

        ext = os.path.splitext(file_path)[1].lower()
        plugin = self._get_plugin_for_extension(ext)
        if not plugin:
            return None

        language_prompts = plugin.get_prompts()
        context_prompt_template = self.plugin_config.get("general_prompts", {}).get(
            "retrieve_context", ""
        )

        formatted_context_prompt = context_prompt_template.format(file_path=file_path)
        relative_path = os.path.relpath(file_path, base_path)

        try:
            req: ReviewRequest = {
                "file_path": file_path,
                "snippet": snippet,
                "retriever_code": qe_code,
                "retriever_docs": qe_docs,
                "context_prompt": formatted_context_prompt,
                "language_prompts": language_prompts,
                "default_prompt_key": "security_review_file",
                "relative_file": relative_path,
                "mode": "file",
                "skip_retrieval": skip_retrieval,
            }
            return self._get_review_graph().review(req)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    def get_code_files(self):
        """
        Return a list of file names in the self.codebase_path folder.
        Evaluate the path with metisignore file, include/exclude paths if requested
        """
        base_path = os.path.abspath(self.codebase_path)
        metisignore_spec = self.load_metisignore()
        include_spec = None
        if self.review_code_include_paths:
            include_spec = pathspec.GitIgnoreSpec.from_lines(
                self.review_code_include_paths
            )
        exclude_spec = None
        if self.review_code_exclude_paths:
            exclude_spec = pathspec.GitIgnoreSpec.from_lines(
                self.review_code_exclude_paths
            )
        file_list = []
        for root, _, files in os.walk(base_path):
            for file in files:
                full_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.code_exts:
                    continue
                rel_path = os.path.relpath(full_path, base_path)
                if metisignore_spec and metisignore_spec.match_file(rel_path):
                    continue
                if include_spec and not include_spec.match_file(rel_path):
                    continue
                if exclude_spec and exclude_spec.match_file(rel_path):
                    continue
                file_list.append(full_path)
        return file_list

    def review_code(self):
        """
        Iterate all supported code files under `codebase_path` and yield
        per-file review results. Uses a thread pool and continues on errors.
        Skips vector retrieval since we're reviewing the entire codebase.
        """
        files = self.get_code_files()
        if not files:
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self.review_file, path, True): path for path in files
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Error reviewing file {path}: {e}")
                    yield None
                    continue
                if result:
                    yield result
                else:
                    yield None

    def review_code_batched(self, batch_token_target=30000, files=None):
        """
        Review all code files, batching small files into single LLM calls.
        Yields per-file result dicts as they complete.
        Files exceeding batch_token_target are reviewed individually.
        Falls back to single-file review on batch parse failure.

        If *files* is provided it is used directly, skipping ``get_code_files()``.
        """
        if files is None:
            files = self.get_code_files()
        if not files:
            return

        base_path = os.path.abspath(self.codebase_path)
        review_graph = self._get_review_graph()

        # Read all files in parallel and measure tokens.
        def _read_and_measure(path):
            snippet = read_file_content(path)
            if not snippet:
                return None
            ext = os.path.splitext(path)[1].lower()
            plugin = self._get_plugin_for_extension(ext)
            if not plugin:
                return None
            return {
                "file_path": path,
                "relative_file": os.path.relpath(path, base_path),
                "snippet": snippet,
                "tokens": count_tokens(snippet),
                "plugin": plugin,
            }

        file_entries = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as io_pool:
            for entry in io_pool.map(_read_and_measure, files):
                if entry is None:
                    yield None
                else:
                    file_entries.append(entry)

        # Separate large files (review individually) from small files (batch)
        large_files = [e for e in file_entries if e["tokens"] > batch_token_target]
        small_files = [e for e in file_entries if e["tokens"] <= batch_token_target]

        # Group small files into batches by plugin
        batches = []
        # Group by plugin name for consistent language_prompts
        by_plugin = {}
        for entry in small_files:
            key = entry["plugin"].get_name()
            by_plugin.setdefault(key, []).append(entry)

        for plugin_name, entries in by_plugin.items():
            current_batch = []
            current_tokens = 0
            for entry in entries:
                if current_tokens + entry["tokens"] > batch_token_target and current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                current_batch.append(entry)
                current_tokens += entry["tokens"]
            if current_batch:
                batches.append(current_batch)

        def _run_batch(batch):
            plugin = batch[0]["plugin"]
            language_prompts = plugin.get_prompts()
            try:
                results = review_graph.review_batch(batch, language_prompts)
                if results is not None:
                    return results
            except Exception as e:
                logger.warning("Batch review failed, falling back to single-file: %s", e)
            # Fallback: review each file individually using already-read snippets.
            fallback_results = []
            for entry in batch:
                try:
                    req: ReviewRequest = {
                        "file_path": entry["file_path"],
                        "snippet": entry["snippet"],
                        "retriever_code": None,
                        "retriever_docs": None,
                        "context_prompt": "",
                        "language_prompts": language_prompts,
                        "default_prompt_key": "security_review_file",
                        "relative_file": entry["relative_file"],
                        "mode": "file",
                        "skip_retrieval": True,
                    }
                    result = review_graph.review(req)
                    fallback_results.append(result)
                except Exception as e2:
                    logger.error("Single-file fallback failed for %s: %s", entry["file_path"], e2)
                    fallback_results.append(None)
            return fallback_results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batches and large single files
            future_to_work = {}
            for batch in batches:
                f = executor.submit(_run_batch, batch)
                future_to_work[f] = ("batch", batch)
            for entry in large_files:
                f = executor.submit(self.review_file, entry["file_path"], True)
                future_to_work[f] = ("single", entry)

            for future in as_completed(future_to_work):
                work_type, work_data = future_to_work[future]
                try:
                    result = future.result()
                except Exception as e:
                    if work_type == "batch":
                        logger.error("Batch review error: %s", e)
                        for _ in work_data:
                            yield None
                    else:
                        logger.error("File review error for %s: %s", work_data["file_path"], e)
                        yield None
                    continue

                if work_type == "batch":
                    # result is a list of per-file dicts
                    if result:
                        for r in result:
                            yield r
                    else:
                        for _ in work_data:
                            yield None
                else:
                    yield result if result else None

    def review_patch(self, patch_file):
        """
        Reviews a patch/diff file by processing each file change.
        """
        qe_code, qe_docs = self._init_and_get_query_engines()
        patch_text = read_file_content(patch_file)
        try:
            diff = unidiff.PatchSet.from_string(patch_text)
            logger.info("Parsed the patch file successfully.")
        except Exception as e:
            logger.error(f"Error parsing patch file: {e}")
            return {"reviews": [], "overall_changes": ""}
        base_path = os.path.abspath(self.codebase_path)
        metisignore_spec = self.load_metisignore()
        context_prompt_tmpl = self.plugin_config.get("general_prompts", {}).get(
            "retrieve_context", ""
        )
        summary_chain = build_summary_chain(self.llm_provider)

        # Collect reviewable diffs (lightweight filtering before thread pool).
        work_items = []
        for file_diff in diff:
            if file_diff.is_removed_file or file_diff.is_binary_file:
                continue
            abs_path = (
                file_diff.path
                if os.path.isabs(file_diff.path)
                else os.path.join(base_path, file_diff.path)
            )
            relative_path = os.path.relpath(abs_path, base_path)
            if metisignore_spec and metisignore_spec.match_file(relative_path):
                continue
            ext = os.path.splitext(file_diff.path)[1].lower()
            plugin = self._get_plugin_for_extension(ext)
            if not plugin:
                continue
            original_content = read_file_content(abs_path)
            snippet = process_diff_file(
                self.codebase_path, file_diff, self.max_token_length,
                original_content=original_content,
            )
            if not snippet:
                continue
            work_items.append((file_diff, abs_path, relative_path, plugin, snippet, original_content))

        def _review_one(item):
            file_diff, abs_path, relative_path, plugin, snippet, original_content = item
            formatted_context = context_prompt_tmpl.format(file_path=file_diff.path)
            language_prompts = plugin.get_prompts()
            try:
                req: ReviewRequest = {
                    "file_path": abs_path,
                    "snippet": snippet,
                    "retriever_code": qe_code,
                    "retriever_docs": qe_docs,
                    "context_prompt": formatted_context,
                    "language_prompts": language_prompts,
                    "default_prompt_key": "security_review",
                    "relative_file": relative_path,
                    "mode": "patch",
                    "original_file": original_content or "",
                }
                review_dict = self._get_review_graph().review(req)
            except Exception as e:
                logger.error(f"Error processing review for {file_diff.path}: {e}")
                return None, None

            if not review_dict:
                return None, None

            issues = "\n".join(
                issue.get("issue", "") for issue in review_dict.get("reviews", [])
            )
            summary_prompt = apply_custom_guidance(
                language_prompts["snippet_security_summary"],
                self.custom_prompt_text,
                self.custom_guidance_precedence,
            )
            changes_summary = summarize_changes(
                self.llm_provider, file_diff.path, issues, summary_prompt,
                chain=summary_chain,
            )
            return review_dict, changes_summary

        file_reviews = []
        overall_summaries = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for review_dict, changes_summary in executor.map(_review_one, work_items):
                if review_dict:
                    file_reviews.append(review_dict)
                if changes_summary:
                    overall_summaries.append(changes_summary)

        overall_changes = "\n\n".join(overall_summaries)
        return {"reviews": file_reviews, "overall_changes": overall_changes}

    def update_index(self, patch_text):
        """
        Updates the existing index by comparing two git commits.
        """
        try:
            patch_set = unidiff.PatchSet.from_string(patch_text)
            logger.info("Parsed the provided patch string successfully.")
        except Exception as e:
            raise ParsingError(f"Error parsing patch string: {e}")
        self.vector_backend.init()
        storage_context_code, storage_context_docs = (
            self.vector_backend.get_storage_contexts()
        )

        index_code = VectorStoreIndex.from_vector_store(
            self.vector_backend.vector_store_code,
            storage_context=storage_context_code,
            embed_model=self.llm_provider.get_embed_model_code(),
        )
        index_docs = VectorStoreIndex.from_vector_store(
            self.vector_backend.vector_store_docs,
            storage_context=storage_context_docs,
            embed_model=self.llm_provider.get_embed_model_docs(),
        )

        doc_splitter = self._get_doc_splitter()

        for diff_file in patch_set:
            if diff_file.is_binary_file:
                continue
            doc_id = os.path.join(
                os.path.basename(os.path.abspath(self.codebase_path)), diff_file.path
            )
            ext = os.path.splitext(doc_id)[1].lower()
            target_index = (
                index_code
                if ext in self._get_all_supported_code_extensions()
                else index_docs
            )

            if diff_file.is_removed_file:
                target_index.delete_ref_doc(doc_id, delete_from_docstore=True)
            else:
                file_path = os.path.join(self.codebase_path, diff_file.path)
                file_content = read_file_content(file_path)
                if not file_content and diff_file.is_added_file:
                    file_content = extract_content_from_diff(diff_file)
                if not file_content:
                    logger.warning("No content available for %s", diff_file.path)
                    continue
                doc = Document(
                    text=file_content,
                    metadata={"file_name": diff_file.path},
                    id_=doc_id,
                )

                if diff_file.is_added_file:
                    if ext in self._get_all_supported_code_extensions():
                        plugin = self._get_plugin_for_extension(ext)
                        if not plugin:
                            continue
                        splitter = self._get_splitter_cached(plugin)
                        try:
                            nodes = splitter.get_nodes_from_documents([doc])
                        except Exception as e:
                            logger.warning(
                                f"Could not parse code with language {plugin.get_name()} for file {doc.id_} (ext {ext}): {e}"
                            )
                            continue
                    else:
                        nodes = doc_splitter.get_nodes_from_documents([doc])
                    target_index.insert_nodes(nodes)
                else:
                    target_index.refresh_ref_docs([doc])
                target_index.docstore.set_document_hash(doc.id_, doc.hash)
        logger.info("Index update complete based on the provided patch diff.")

    def _init_and_get_query_engines(self):
        self.vector_backend.init()
        qe_code, qe_docs = self.vector_backend.get_query_engines(
            self.llm_provider,
            self.similarity_top_k,
            self.response_mode,
        )
        if not qe_code or not qe_docs:
            raise QueryEngineInitError()
        return qe_code, qe_docs
