"""Speaker attribution: fuse Soniox diarization with ASD face identity."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---- Speaker-fusion weights (tweak to bias the binding algorithm) ----
#
# Each Soniox speaker id (e.g. "1", "2") accumulates a per-candidate
# score table. The wearer is the prior: every newly-seen Soniox spk_id
# starts with WEARER_PRIOR weight on USER_NAME, because the glasses
# mic primarily picks up the wearer regardless of who's on screen.
# When ASD reports a named face during a Soniox segment, that name
# gains W_ASD_VOTE + W_ASD_DURATION*seconds. When ASD reports NO
# speaking face, USER_NAME gains W_SILENT_FACE_USER*seconds. Other
# face candidates take a small W_OTHER_FACE_PENALTY drag per second so
# a clear winner pulls ahead faster — but the wearer is exempt from
# this penalty, since "ASD saw face X speaking" is weak evidence the
# wearer wasn't (both can be true; the wearer's mic always hears them).
# Once the top candidate beats the runner-up by MIN_LOCK_MARGIN AND
# crosses MIN_LOCK_SCORE, the spk_id → name binding is locked for the
# rest of the session and the pending gallery entry for that name is
# promoted.
W_ASD_VOTE = 1.5
W_ASD_DURATION = 0.8
W_SILENT_FACE_USER = 1.0
W_OTHER_FACE_PENALTY = 0.25
# Continuous wearer credit per second of Soniox speech (applied on every
# segment regardless of ASD result). Reflects the prior that the glasses
# mic primarily picks up the wearer; without it, a chronically visible
# guest face whose lips move during listening can outpace the wearer.
W_WEARER_BASELINE = 0.4
# Minimum Soniox segment length (in seconds) for ASD evidence to count.
# Shorter than this, treat as too noisy to attribute reliably (filler
# words, "mhm", brief reactions). Wearer baseline still applies.
MIN_SEGMENT_S_FOR_ASD = 0.4
WEARER_PRIOR = 0.5
MIN_LOCK_SCORE = 2.5
MIN_LOCK_MARGIN = 1.0
# A spk_id must have accumulated this many seconds of total speech
# before we'll lock it to an identity. Soniox's v4 RT model spends the
# first few seconds of a session emitting placeholder speaker IDs that
# may not yet differentiate distinct voices; locking too early can
# cement two real voices to one identity. 5 s is conservative — the
# user will see provisional labels until then.
MIN_LOCK_TOTAL_S = 5.0


class SpeakerFusionMixin:
    """Fuse Soniox per-token speaker ids with ASD face-side identity."""

    asd: Any
    user_name: str

    def _init_fusion_state(self) -> None:
        # spk_id -> locked identity name (cleared on every reconnect
        # because Soniox restarts speaker numbering).
        self._spk_to_identity: dict[str, str] = {}
        # spk_id -> {candidate_name -> running fusion score}
        self._spk_score: dict[str, dict[str, float]] = {}
        # spk_id -> total ms of speech and total ms with no speaking face.
        self._spk_total_ms: dict[str, int] = {}
        self._spk_silent_face_ms: dict[str, int] = {}
        # spk_id -> provisional display label assigned in order of
        # first appearance. The first Soniox spk we see is treated as
        # the wearer; later distinct spks become 'Speaker 2', etc.
        # This guarantees that distinct Soniox voices always render as
        # distinct labels in the transcript, even before ASD binds a
        # face name to them.
        self._spk_provisional: dict[str, str] = {}

    def _clear_fusion_state(self) -> None:
        self._spk_to_identity.clear()
        self._spk_score.clear()
        self._spk_total_ms.clear()
        self._spk_silent_face_ms.clear()
        self._spk_provisional.clear()

    def _stream_ms_to_mono(self, stream_ms: float) -> float | None:
        """Convert a Soniox stream_ms timestamp to a monotonic() time.
        Assumes audio ingestion is real-time (it is — aiortc streams)."""
        anchor = self._clock_anchor
        if anchor is None:
            return None
        anchor_ms, anchor_mono = anchor
        return anchor_mono - (anchor_ms - stream_ms) / 1000.0

    def _bind_soniox_speaker_to_identity(
        self,
        spk: str,
        asd_name: str | None,
        dur_ms: int,
    ) -> str | None:
        """Update the running fusion score table for Soniox speaker
        `spk` from one classified segment, then check whether the
        winning candidate has crossed the lock-in gates. Returns the
        locked name if the lock just fired this call, else None.

        `asd_name = None` means ASD ran but saw no speaking face during
        the window — strong wearer evidence (the glasses mic hears the
        wearer even when nobody is on camera)."""
        self._spk_total_ms[spk] = self._spk_total_ms.get(spk, 0) + dur_ms
        sec = dur_ms / 1000.0
        is_first = spk not in self._spk_score
        table = self._spk_score.setdefault(spk, {})
        # Wearer prior: small head start on first observation of a new
        # spk_id. Kept small (0.5) so a real guest with consistent ASD
        # evidence can overtake within ~1 segment — when Soniox gives
        # us distinct spk_ids per voice, that's the authoritative
        # diarization signal and we should let it differentiate.
        if is_first:
            table[self.user_name] = WEARER_PRIOR

        # Continuous wearer baseline: a slow, always-on drip so the
        # wearer holds the line on segments where ASD is uncertain.
        # Smaller than the per-segment ASD vote so a real speaking face
        # (consistent ASD positive over a real-length segment) wins.
        table[self.user_name] = table.get(self.user_name, 0.0) + W_WEARER_BASELINE * sec

        if asd_name is None:
            self._spk_silent_face_ms[spk] = self._spk_silent_face_ms.get(spk, 0) + dur_ms
            table[self.user_name] = table.get(self.user_name, 0.0) + W_SILENT_FACE_USER * sec
        elif sec >= MIN_SEGMENT_S_FOR_ASD:
            table[asd_name] = table.get(asd_name, 0.0) + W_ASD_VOTE + W_ASD_DURATION * sec
            # Drag every OTHER face candidate so a clear winner pulls
            # ahead, but exempt the wearer: "ASD saw face X" is not
            # evidence "the wearer wasn't speaking" — both can be true,
            # and we'd rather under-credit a guest than steal the
            # wearer's identity.
            for other in list(table.keys()):
                if other != asd_name and other != self.user_name:
                    table[other] -= W_OTHER_FACE_PENALTY * sec
        # else: short segment with an ASD candidate — too noisy, skip
        # face evidence; only the wearer baseline above counts.

        if spk in self._spk_to_identity:
            return None  # already locked

        if not table:
            return None
        # Hold the lock until the spk_id has enough total speech to be
        # meaningful. Avoids locking based on Soniox's cold-start
        # placeholder speaker tags before voices are differentiated.
        if self._spk_total_ms.get(spk, 0) / 1000.0 < MIN_LOCK_TOTAL_S:
            return None
        sorted_scores = sorted(table.values(), reverse=True)
        top_name, top_score = max(table.items(), key=lambda kv: kv[1])
        second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        if top_score >= MIN_LOCK_SCORE and (top_score - second) >= MIN_LOCK_MARGIN:
            self._spk_to_identity[spk] = top_name
            logger.debug(
                "[soniox-diag] LOCK spk=%s -> %s (score=%.2f, runner-up=%.2f)",
                spk,
                top_name,
                top_score,
                second,
            )
            # Promote the pending gallery entry for a face we've now
            # confirmed is actually audible. No-op for the wearer.
            if self.asd is not None and top_name != self.user_name:
                try:
                    self.asd.confirm_identity_for_name(top_name)
                except Exception as e:
                    logger.error(
                        "[soniox] confirm_identity_for_name(%r) failed: %s",
                        top_name,
                        e,
                    )
            return top_name
        return None

    def _provisional_label_for_spk(self, spk: str) -> str:
        """Display label for an unbound Soniox speaker id.

        Critically: this is NOT the wearer name. The wearer fallback
        only applies when Soniox has no speaker tag at all. Once
        Soniox differentiates two voices (spk='1' vs spk='2'), we MUST
        render them as distinguishable labels so the user sees that
        diarization is working — even before ASD has put a face name
        on each spk_id. The first Soniox-tagged spk we see is treated
        as the wearer (the wearer almost always speaks first into
        their own glasses); subsequent new spk_ids render as
        'Speaker 2', 'Speaker 3', etc."""
        # Cache by spk_id so labels are stable within a session.
        cached = self._spk_provisional.get(spk)
        if cached is not None:
            return cached
        # First spk we observe → wearer; everyone else gets a numeric
        # provisional label keyed by order of appearance.
        if not self._spk_provisional:
            label = self.user_name
        else:
            label = f"Speaker {len(self._spk_provisional) + 1}"
        self._spk_provisional[spk] = label
        return label

    def _classify_speaker(
        self,
        start_ms: int,
        end_ms: int,
        spk: str | None,
    ) -> str:
        """Attribute a finalized segment by FUSING the Soniox speaker
        id (audio-side diarization) with ASD's face-side identity.

        - If Soniox gave us no speaker (`spk is None` or empty/"0"),
          fall back to the wearer label — there's nothing to bind to.
        - If `spk` is already bound to a name, that's authoritative.
        - Otherwise ASD names the most-speaking face during the window
          and feeds the running fusion score for `spk`. While the
          binding is unsettled, render a Soniox-derived provisional
          label so distinct voices show as distinct in the transcript
          even before face attribution catches up."""
        if not spk or spk == "0" or end_ms <= start_ms:
            return self.user_name

        bound = self._spk_to_identity.get(spk)
        if bound is not None:
            return bound

        name: str | None = None
        if self.asd is not None:
            t_start = self._stream_ms_to_mono(start_ms)
            t_end = self._stream_ms_to_mono(end_ms)
            if t_start is not None and t_end is not None:
                try:
                    name = self.asd.who_spoke_name(t_start, t_end, min_samples=2, prefer_named=True)
                except Exception as e:
                    logger.warning("[soniox] asd lookup failed: %s", e)
                    name = None

        self._bind_soniox_speaker_to_identity(spk, name, end_ms - start_ms)

        if spk in self._spk_to_identity:
            return self._spk_to_identity[spk]

        # Provisional label keyed off Soniox's spk_id so distinct
        # voices ALWAYS render as distinct labels, even before ASD
        # binds a face. Prefer ASD's name when it has one (so a real
        # face shows up sooner than the generic 'Speaker N').
        provisional = self._provisional_label_for_spk(spk)
        chosen = name or provisional

        # Always-on diagnostic so we can see why a segment got the
        # label it got. Drop the `[soniox-diag]` prefix once tuning is
        # settled.
        logger.debug(
            "[soniox-diag] seg_dur=%.2fs spk=%s asd=%r prov=%r table=%s -> %r",
            (end_ms - start_ms) / 1000.0,
            spk,
            name,
            provisional,
            {k: round(v, 2) for k, v in (self._spk_score.get(spk) or {}).items()},
            chosen,
        )
        return chosen
