from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class VaultNote:
    path: str
    title: str
    folder: str
    content: str
    tokens: List[str]
    length: int
    modified_time: float
    somatic_valence: float = 0.5  # 0.0 to 1.0 (reinforced by dopamine on task success)


@dataclass
class MemorySearchResult:
    path: str
    title: str
    score: float
    snippet: str
    somatic_valence: float


class MemoryVaultEngine:
    """
    High-Performance Memory Indexing & Retrieval Engine for Aira.
    Connects to Obsidian Vault (C:\\Users\\WaveWalker\\Aira-Vault or local workspace).
    Implements:
    - In-memory BM25 inverted index for sub-millisecond lexical search
    - Somatic affective weighting (high-dopamine notes prioritized)
    - Sub-folder hierarchical caching
    - Episodic memory logging and checkpoint persistence
    """

    DEFAULT_VAULT_PATH = Path(r"C:\Users\WaveWalker\Aira-Vault")

    def __init__(self, vault_path: str | Path | None = None):
        self.vault_path = Path(vault_path) if vault_path else self.DEFAULT_VAULT_PATH
        self.notes: Dict[str, VaultNote] = {}
        self.inverted_index: Dict[str, Set[str]] = {}
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len: float = 1.0
        self.last_index_time: float = 0.0

        # Build index if vault exists
        if self.vault_path.exists():
            self.refresh_index()

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
        return words

    def refresh_index(self) -> int:
        """Indexes all markdown notes in the vault with token frequency metrics."""
        if not self.vault_path.exists():
            return 0

        indexed_count = 0
        total_len = 0
        new_notes: Dict[str, VaultNote] = {}
        new_inv_idx: Dict[str, Set[str]] = {}
        new_doc_freq: Dict[str, int] = {}

        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden folders like .git or .obsidian
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if not file.endswith(".md"):
                    continue
                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(self.vault_path))
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                tokens = self._tokenize(content)
                doc_len = len(tokens)
                total_len += doc_len
                folder = full_path.parent.name
                title = full_path.stem

                # Check if somatic valence exists in metadata
                valence = 0.5
                if "dopamine_verified" in content:
                    valence = 0.95

                note = VaultNote(
                    path=rel_path,
                    title=title,
                    folder=folder,
                    content=content,
                    tokens=tokens,
                    length=doc_len,
                    modified_time=full_path.stat().st_mtime,
                    somatic_valence=valence
                )
                new_notes[rel_path] = note

                # Inverted index
                unique_tokens = set(tokens)
                for tok in unique_tokens:
                    new_inv_idx.setdefault(tok, set()).add(rel_path)
                    new_doc_freq[tok] = new_doc_freq.get(tok, 0) + 1

                indexed_count += 1

        self.notes = new_notes
        self.inverted_index = new_inv_idx
        self.doc_freq = new_doc_freq
        self.avg_doc_len = (total_len / indexed_count) if indexed_count > 0 else 1.0
        self.last_index_time = time.time()
        return indexed_count

    def search(self, query: str, top_k: int = 5) -> List[MemorySearchResult]:
        """
        Sub-millisecond BM25 ranking augmented by Somatic Marker valence.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.notes:
            return []

        # Candidate notes that contain at least one query token
        candidates: Set[str] = set()
        for tok in q_tokens:
            if tok in self.inverted_index:
                candidates.update(self.inverted_index[tok])

        if not candidates:
            return []

        n_docs = len(self.notes)
        k1 = 1.5
        b = 0.75
        scored: List[Tuple[str, float]] = []

        for path in candidates:
            note = self.notes[path]
            bm25_score = 0.0
            note_token_counts = {}
            for t in note.tokens:
                note_token_counts[t] = note_token_counts.get(t, 0) + 1

            for q_tok in q_tokens:
                df = self.doc_freq.get(q_tok, 0)
                if df == 0:
                    continue
                # Standard Robertson-Spärck Jones IDF
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = note_token_counts.get(q_tok, 0)
                denom = tf + k1 * (1.0 - b + b * (note.length / self.avg_doc_len))
                bm25_score += idf * ((tf * (k1 + 1.0)) / denom)

            # Title match boost
            title_tokens = set(self._tokenize(note.title))
            for q_tok in q_tokens:
                if q_tok in title_tokens:
                    bm25_score += 2.5

            # Somatic Valence boost (high-dopamine verified memories ranked higher)
            final_score = bm25_score * (0.8 + 0.4 * note.somatic_valence)
            scored.append((path, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []

        for path, score in scored[:top_k]:
            note = self.notes[path]
            # Extract most relevant excerpt centered on matched terms, stripping frontmatter
            clean_body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", note.content, flags=re.DOTALL)
            matched_idx = -1
            for q_tok in q_tokens:
                idx = clean_body.lower().find(q_tok.lower())
                if idx != -1 and (matched_idx == -1 or idx < matched_idx):
                    matched_idx = idx

            if matched_idx != -1:
                start_pos = max(0, matched_idx - 60)
                end_pos = min(len(clean_body), matched_idx + 240)
                snippet = clean_body[start_pos:end_pos].strip().replace("\n", " ")
            else:
                snippet = clean_body[:300].strip().replace("\n", " ")

            results.append(
                MemorySearchResult(
                    path=note.path,
                    title=note.title,
                    score=round(score, 3),
                    snippet=snippet,
                    somatic_valence=note.somatic_valence
                )
            )

        return results

    def record_episodic_memory(self, title: str, summary: str, details: str, dopamine_verified: bool = True) -> str:
        """
        Persists a solved problem or important insight into the episodic vault.
        """
        today_str = time.strftime("%Y-%m-%d")
        dest_dir = self.vault_path / "01 - Daily Notes"
        if not dest_dir.exists():
            dest_dir = self.vault_path / "Episodic_Memories"
        dest_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        filename = f"{today_str}_{safe_title}.md"
        target_path = dest_dir / filename

        valence_tag = "[dopamine_verified: True]" if dopamine_verified else "[status: unverified]"
        entry = (
            f"---\n"
            f"status: active\n"
            f"project: episodic\n"
            f"type: log\n"
            f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"somatic_valence: {0.95 if dopamine_verified else 0.50:.2f}\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"- Date: {today_str}\n"
            f"- Somatic Marker: {valence_tag}\n\n"
            f"## Executive Summary\n{summary}\n\n"
            f"## Technical Details\n{details}\n"
        )

        try:
            target_path.write_text(entry, encoding="utf-8")
            self.refresh_index()
            return str(target_path)
        except Exception as exc:
            return f"Error writing episodic memory: {exc}"

    def record_chat_session(
        self,
        session_id: str,
        topic: str,
        topic_description: str,
        turns: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
        important_notes: Optional[List[str]] = None,
        decisions_made: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        user_trust: float = 0.5,
        somatic_valence: float = 0.5,
    ) -> str:
        """
        Persists an entire conversation session into the vault across all chats:
        - Writes markdown note with YAML frontmatter to Conversations/
        - Appends session reference to today's Daily Note
        - Re-indexes memory vault immediately for cross-chat retrieval
        """
        important_notes = important_notes or []
        decisions_made = decisions_made or []
        keywords = keywords or []

        date_str = time.strftime("%Y-%m-%d", time.localtime(start_time))
        time_start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        time_end_str = time.strftime("%H:%M:%S", time.localtime(end_time))
        safe_topic = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")
        if not safe_topic:
            safe_topic = "Session"

        time_slug = time.strftime("%H%M", time.localtime(start_time))
        filename = f"{date_str}_{time_slug}_{safe_topic}.md"

        conv_dir = self.vault_path / "Conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        target_path = conv_dir / filename

        # Extract topics / tags for frontmatter
        clean_tags = [re.sub(r"[^\w-]", "", kw.lower()) for kw in keywords if kw]
        if not clean_tags:
            clean_tags = ["general-chat"]

        notes_bullets = "\n".join(f"- {n}" for n in important_notes) if important_notes else "- No specific constraints logged."
        decisions_bullets = "\n".join(f"- {d}" for d in decisions_made) if decisions_made else "- Standard exploratory dialogue."

        # Format transcript lines
        transcript_lines = []
        for t in turns:
            sender = t.get("sender", "Unknown")
            ts = t.get("timestamp", "")
            text = t.get("text", "").replace("\n", "\n  > ")
            mode = t.get("cognitive_mode", "")
            mode_tag = f" [{mode.upper()}]" if mode else ""
            transcript_lines.append(f"- **[{ts}] {sender}{mode_tag}**:\n  > {text}\n")

        transcript_body = "\n".join(transcript_lines)

        content = (
            f"---\n"
            f"status: active\n"
            f"project: aira-chat\n"
            f"type: log\n"
            f"created: {time_start_str}\n"
            f"session_id: {session_id}\n"
            f"topics: [{', '.join(clean_tags[:8])}]\n"
            f"somatic_valence: {somatic_valence:.2f}\n"
            f"---\n\n"
            f"# Chat Session: {topic}\n\n"
            f"- **Timeline**: {time_start_str} to {time_end_str}\n"
            f"- **User Trust Level**: {user_trust * 100:.1f}%\n"
            f"- **Somatic Valence**: {somatic_valence:.2f}\n\n"
            f"## Topic Overview & Executive Description\n"
            f"{topic_description}\n\n"
            f"## Important Notes & Key Insights\n"
            f"{notes_bullets}\n\n"
            f"## Decisions Made\n"
            f"{decisions_bullets}\n\n"
            f"## Complete Dialogue Log\n"
            f"{transcript_body}\n"
        )

        try:
            target_path.write_text(content, encoding="utf-8")

            # Update / Link to Today's Daily Note
            self._link_to_daily_note(
                date_str=date_str,
                time_slug=time_slug,
                topic=topic,
                topic_description=topic_description,
                notes=important_notes,
                decisions=decisions_made,
                conv_filename=filename
            )

            # Re-index to make searchable immediately
            self.refresh_index()
            return str(target_path)
        except Exception as exc:
            return f"Error recording chat session: {exc}"

    def _link_to_daily_note(
        self,
        date_str: str,
        time_slug: str,
        topic: str,
        topic_description: str,
        notes: List[str],
        decisions: List[str],
        conv_filename: str
    ) -> None:
        """Appends session recap with wikilink to today's daily note in 01 - Daily Notes."""
        month_folder_name = time.strftime("%m - %B %Y")
        daily_notes_dir = self.vault_path / "01 - Daily Notes" / month_folder_name
        daily_notes_dir.mkdir(parents=True, exist_ok=True)
        daily_file = daily_notes_dir / f"{date_str}.md"

        session_entry = (
            f"\n\n## Session — {time_slug[:2]}:{time_slug[2:]}: {topic}\n\n"
            f"### Overview\n"
            f"{topic_description}\n\n"
            f"### Key Notes & Decisions\n"
        )
        if notes:
            session_entry += "\n".join(f"- [Note] {n}" for n in notes) + "\n"
        if decisions:
            session_entry += "\n".join(f"- [Decision] {d}" for d in decisions) + "\n"
        if not notes and not decisions:
            session_entry += "- Continuous cognitive engagement.\n"

        session_entry += f"\n- **Vault Link**: [[Conversations/{conv_filename}|Full Conversation Transcript]]\n"

        if daily_file.exists():
            with daily_file.open("a", encoding="utf-8") as f:
                f.write(session_entry)
        else:
            # Initialize with proper YAML frontmatter
            header = (
                f"---\n"
                f"status: active\n"
                f"project: personal\n"
                f"type: log\n"
                f"created: {date_str}\n"
                f"---\n\n"
                f"# Daily Note — {date_str}\n"
            )
            daily_file.write_text(header + session_entry, encoding="utf-8")

    def get_chat_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns recent chat sessions with timestamps, topics, and important notes."""
        conv_dir = self.vault_path / "Conversations"
        if not conv_dir.exists():
            return []

        files = list(conv_dir.glob("*.md"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        sessions = []
        for f in files[:limit]:
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
                title_match = re.search(r"^#\s+(?:Chat Session:\s*)?(.+)$", txt, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else f.stem
                created_match = re.search(r"^created:\s*(.+)$", txt, re.MULTILINE)
                created = created_match.group(1).strip() if created_match else ""

                # Extract overview
                overview_match = re.search(
                    r"## Topic Overview & Executive Description\n(.*?)(?=\n##|\Z)",
                    txt,
                    re.DOTALL
                )
                overview = overview_match.group(1).strip() if overview_match else ""

                # Extract notes
                notes_match = re.search(
                    r"## Important Notes & Key Insights\n(.*?)(?=\n##|\Z)",
                    txt,
                    re.DOTALL
                )
                raw_notes = notes_match.group(1).strip() if notes_match else ""
                notes = [line.lstrip("- ").strip() for line in raw_notes.split("\n") if line.strip().startswith("-")]

                sessions.append({
                    "filename": f.name,
                    "path": str(f),
                    "title": title,
                    "created": created,
                    "overview": overview[:200],
                    "notes": notes,
                })
            except Exception:
                continue

        return sessions

    def search_chat_history(self, query: str, top_k: int = 5) -> List[MemorySearchResult]:
        """Searches specifically across past conversation notes."""
        all_results = self.search(query, top_k=top_k * 2)
        conv_results = [r for r in all_results if "Conversations" in r.path or "aira-chat" in r.snippet]
        if not conv_results:
            return all_results[:top_k]
        return conv_results[:top_k]

