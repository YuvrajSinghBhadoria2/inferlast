# FIRST-USER ASK — how to seed inferlast's first real community signal

**Why this exists:** the project cannot become a company on features alone. The make-or-break step
(which the agent honestly cannot fabricate) is **one genuine external human** running inferlast and
reporting what happened. That single report is worth more for credibility than any feature.

## The exact ask (copy-paste to ONE developer you know)

> Hey — building a small CPU-first tool called inferlast and need one honest pair of eyes.
> It profiles which part of an LLM's inference actually eats your time on a plain CPU — no GPU
> needed — and **refuses to call an optimization a win unless the benchmark is statistically real**
> (it caught its own 3x-vs-0.65x result as pure noise). That "is this win actually real?" check is
> the part I think is genuinely useful.
>
> Could you run this one command on your machine and tell me what the report says?
>
> ```bash
> pip install -r requirements.txt
> python scripts/run_all.py --model Qwen/Qwen2.5-0.5B-Instruct
> ```
>
> (10 min, CPU only, ~1 GB model. Repo: https://github.com/YuvrajSinghBhadoria2/inferlast )
> What I want back: did it run, did the verdict make sense to you, and would you trust it?
> No pressure — even "it's confusing" is genuinely useful.

## Why this exact wording

- **Names the ONE novel thing** (trustcheck) — not "an optimizer."
- **Sets low effort + CPU-only + small model** — removes every reason to say no.
- **Asks for a specific, non-onerous output** ("would you trust it?") — lowers the commitment.
- **Invites negative feedback** ("confusing is useful") — honesty is the brand, so welcome it.

## What to do with whatever they say

- **It ran + verdict made sense** → open a GitHub Discussion titled *"First external run: <hardware>"*
  and paste it. Link it from README under a "Ran it? Report it" badge-ish line.
- **It confused them** → fix the confusion (that IS a bug), thank them, log it.
- **They never reply** → move on; ask one other person. Do not nag.

## Rules (honesty discipline)

- Never ask someone to post a fake positive report. Only real runs get posted.
- Post the report verbatim, hardware + environment included, failures included.
- The README "testimonials" section gets only links to real Discussions, never invented quotes.