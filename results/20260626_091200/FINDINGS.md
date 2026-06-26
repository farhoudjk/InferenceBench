# Benchmark Findings

**Run:** `20260626_091200` · Qwen2.5-7B · 1× V100 · strategy `baseline_full_fp16_tp1`
**Coverage:** 4 workloads (chat, code, summarization, reasoning) × 3 traffic rates (2/4/8 req per sec) × 50 requests = **600 requests, 0 errors.**

---

## The basic setup (first principles)

Think of the one GPU as **one cook in a kitchen**.

- Every **request** is an order.
- For each order the cook first **reads** the input, then **writes** the answer one word at a time.
- One cook can only do so much. If orders arrive faster than they finish, they pile up and wait.

Two different clocks matter:

- **Wait-for-first-word** — how long before the answer starts. Driven mostly by **how much it has to read** (input size).
- **Total time** — how long until the answer is done. Driven mostly by **how many words it writes** (output size).

Almost everything below falls out of these two ideas.

---

## Finding 1: The *kind* of work matters most

Wait-for-first-word, at the calmest traffic (2/sec):

```
chat           155 ms    ▏
reasoning      211 ms    ▏
code           337 ms    ▎
summarization  21,066 ms ████████████████████████  (21 seconds!)
```

**Why:** summarization sends a huge chunk of text to read, so the first word is far away. Chat sends a tiny message, so the first word is almost instant. Same cook, same model — the only difference is how much reading there is up front.

---

## Finding 2: First word fast ≠ finished fast

Total time to finish the whole answer, at 2/sec:

```
chat            5.3 s   ███
reasoning      46.7 s   ███████████████████████
code           30.8 s   ███████████████
summarization  52.4 s   ██████████████████████████
```

**Reasoning** gives the first word in 0.2 sec but takes **47 sec** to finish — short question (fast to read), long answer (slow to write). This proves the two clocks are driven by different things: **reading size → first word, writing size → total time.**

---

## Finding 3: More traffic makes everyone slower

One cook + more orders at once = each order gets less attention. Chat, as traffic rises 2 → 4 → 8 per sec:

Wait-for-first-word:
```
2/sec   155 ms   ███████
4/sec   207 ms   █████████
8/sec   426 ms   ████████████████████
```

Words-per-second (how fast the answer streams out):
```
2/sec   26.5 w/s ████████████████████
4/sec   20.6 w/s ███████████████
8/sec   16.6 w/s ████████████
```

**Why:** more orders running together means the cook splits attention. Each one waits longer to start *and* streams out slower. Code shows this even more sharply — its wait-for-first-word jumps from 0.3 sec to **3.6 sec** under heavy traffic.

---

## Finding 4: Summarization caused traffic jams

**Preemptions** = times the cook had to pause a half-done order and set it aside to make room.

```
chat           0
code           0
reasoning      0
summarization  42  ← only here
```

**Why:** the giant summarization inputs fill up the GPU's working memory. When it fills, the server must shove some orders aside temporarily, then return to them. That's also why summarization started quickly 0% of the time at every traffic level — too much to chew on.

---

## Bottom line

1. **The experiment is healthy** — 600/600 requests succeeded, every answer complete, no errors.
2. **Everything behaves as expected** — big input → slow to start, long output → slow to finish, more traffic → slower for all, giant inputs → memory pressure and pauses.
3. **Nothing here is a bug** — these are the real trade-offs of running this model on one V100.

*Caveat (unrelated to results): the "GPU busy %" sensor logged 0% the whole run, which can't be literally true — that one gauge wasn't reading correctly. It does not affect any timing numbers above.*
