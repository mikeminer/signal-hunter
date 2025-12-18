ETH Liquidity Grab Signals – PRO Guide

Advanced Trading Manual
Liquidity Pools • Stop Hunts • Max Pain • Real-Time Alerts

📌 Overview

This system is NOT a classic trading indicator.
It is a real-time detector of forced liquidity events on ETH perpetuals.

It stays silent most of the time and alerts only when the market structure shows stop hunts and max-pain conditions.

Silence = no edge
Alert = attention required

🧠 Core Philosophy

The system does NOT:

Predict price direction

Follow trends

Generate frequent signals

Replace risk management

The system DOES:

Detect liquidity pools (equal highs / equal lows)

Identify stop hunts

Alert during forced liquidation events

Provide timing, not bias

🔎 What Is a Liquidity Grab

A liquidity grab occurs when price moves aggressively into areas where:

Stop losses are clustered

Liquidations are likely

Weak traders are forced out

After the grab:

Price often reverts

Volatility spikes

Risk/reward improves

This system detects those moments only.

📈 Signal Types
🔴 SHORT Signal (Bull Trap)

Conditions:

Equal highs (3–5+ touches)

Liquidity above price

Funding positive or neutral (long crowd)

Price sweeps above the level

Strong rejection OR reclaim below

Result:

SHORT signal after stop hunt

🟢 LONG Signal (Bear Trap)

Conditions:

Equal lows (3–5+ touches)

Liquidity below price

Funding negative or neutral (short crowd)

Price sweeps below the level

Strong rejection OR reclaim above

Result:

LONG signal after stop hunt

📊 Signal Probability (%)

Each signal includes a probability score (0–100%).

This is NOT a prediction, but a quality score based on:

Factor	Description
Touches	Number of hits on the level
Sweep depth	How far price ran stops
Rejection	Wick vs body ratio
Funding	Crowd positioning bias
Reclaim	Conservative confirmation
Probability Interpretation

≥ 75% → Strong signal (priority)

60–74% → Medium quality

< 60% → Informational / discretionary

🔊 Alert System (Siren)

When a valid signal appears:

A continuous siren starts

Visual log is printed

Alarm remains active until action

Controls

ACK → Silence the siren, keep monitoring

STOP → Stop streaming and alarm

⏱️ Cooldown (Anti-Spam)

The cooldown prevents alert flooding during high volatility.

Behavior:

Signals inside cooldown are logged

Siren does NOT retrigger

Next alert waits until cooldown expires

Recommended:

30–90 seconds

🧭 Decision Flowchart
Are there equal highs / lows?
 ├─ No → Stay out
 └─ Yes
     ↓
Did price sweep the level?
 ├─ No → Stay out
 └─ Yes
     ↓
Is there rejection or reclaim?
 ├─ No → Stay out
 └─ Yes
     ↓
Is funding coherent?
 ├─ No → Reduce size
 └─ Yes
     ↓
Is probability ≥ 70%?
 ├─ No → Discretionary
 └─ Yes → ENTER

🛠️ Practical Usage Rules

Best during London / NY sessions

Works best on ETH perpetuals

Use moderate leverage

Enter after or during the stop hunt

Never force trades when system is silent

⚠️ Risk Disclaimer

This tool:

Does not guarantee profits

Does not replace stop-loss discipline

Is a decision support system

Always manage:

Position size

Max daily loss

Emotional exposure

🏷️ Quick Reference (1 Page)
Element	Rule
Signal	Only after stop hunt
Probability	≥75% = strong
Siren	Active liquidity event
ACK	Silence alarm
STOP	Stop system
Cooldown	Anti-spam
Silence	No edge
