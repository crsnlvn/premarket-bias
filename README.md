# Pre-Market Bias

A free daily NQ / ES read for the New York morning. It answers one question: bullish or bearish into the open, and why. The read draws on dealer gamma, charm, vanna, options flow, volatility and macro.

GitHub builds the report on weekday mornings before the open, on its servers. Your phone just opens the page. There is nothing to pay for and nothing running on your devices.

---

## One-time setup (about 15 minutes, easiest on a PC)

**1. Make a GitHub account** at github.com if you don't have one. The free plan is all you need.

**2. Create a public repository.**
Click **+ → New repository**. Name it something like `premarket-bias`, choose **Public**, and click **Create repository**.
> It has to be public, because public repos get free automation minutes and free Pages hosting.

**3. Upload these files.**
On the new repo page, click **uploading an existing file**. Unzip this project, then drag *everything inside the folder* into the browser window. That includes the hidden `.github` folder.
> Windows hides folders that start with a dot. If `.github` doesn't upload, do step 3b.

**3b. Only if `.github` didn't upload:** click **Add file → Create new file** and type `.github/workflows/morning.yml` as the name. The slashes create the folders. Paste in the contents of `morning.yml` from the zip, then click **Commit**.

**4. Let the robot save its work.**
Go to **Settings → Actions → General → Workflow permissions**, choose **Read and write permissions**, and click **Save**.

**5. Turn on the website.**
Go to **Settings → Pages**. Under *Build and deployment*, set Source to **Deploy from a branch**. Set Branch to **main** and the folder to **/docs**, then click **Save**.
After a minute or two it shows your link, something like `https://YOURNAME.github.io/premarket-bias/`.

**6. Run it once now.**
Go to the **Actions** tab. If asked, click **I understand my workflows, enable them**. Then pick **Morning bias report → Run workflow**. It takes about a minute, and the page fills in with the latest data.

**7. Put it on your phone.**
- **iPhone:** open the link in **Safari**, tap **Share**, then **Add to Home Screen**.
- **Android:** open the link in **Chrome**, tap **⋮**, then **Install app** (or **Add to Home screen**).

It now opens full-screen like any other app.

---

## Every morning

Nothing to do. The report builds on weekdays around **8:10 AM ET**, refreshes around **9:10 AM ET**, and GitHub can start either run a few minutes late. Open the app and pull it up. A green dot means today's report. An amber dot means the morning run hasn't landed yet.

**From your phone**, the GitHub mobile app lets you:
- re-run it at any time: **Actions → Morning bias report → Run workflow**, for example right after an 8:30 data print
- see if a run failed: a red ✕ appears in Actions
- edit thresholds and weights: open `scripts/build_report.py`, tap the pencil, and commit

---

## What goes into the score

| Group | Signal | Logic |
|---|---|---|
| Dealer | Gamma flip | Price above the zero-gamma level means dealers buy dips and sell rips. Below it, their hedging chases moves. |
| Dealer | Gamma tilt | Call vs put gamma balance |
| Dealer | Wall range | Where price sits between the put wall (support) and the call wall (resistance) |
| Dealer | Flip into close | Whether the flip drifts away from or toward price as today's options expire |
| Dealer | Charm | Dealer hedge buying or selling forced purely by time decay into 4 PM |
| Dealer | Vanna | Hedge flow forced by the overnight change in implied vol (VXN / VIX) |
| Flow | Put/call premium | Prior session's short-dated premium by side |
| Vol | VXN / VIX overnight | Fear rising or bleeding out |
| Vol | Term structure | VIX ÷ VIX3M. Backwardation is a warning. |
| Macro | Overnight futures | Globex move vs settlement |
| Macro | 10-year yield | Rates up hurts long-duration tech hardest |
| Macro | Dollar, crude | Financial conditions and energy shock |

Each signal scores +1, 0 or −1 and is multiplied by its weight. The total is scaled to −1…+1. Conviction drops a notch when a high-impact US release is still ahead. Mega-cap earnings are flagged.

**Free data sources:** CBOE delayed option chains (NDX + QQQ, SPX + SPY), Yahoo Finance, the ForexFactory calendar feed, and the Nasdaq earnings calendar. If one source is down that morning, its signals drop out and the page lists it under *Data notes*.

**Track record:** each report is graded the next day on the NQ/ES move from 9:30 to 12:00 ET, so you can see whether the model earns its keep.

### Honest caveats
- Option data is 15 minutes delayed. Before the open, OI is last night's settled open interest, which is exactly what the morning read needs.
- Dealer positioning is a **model**. It assumes dealers are long calls and short puts, the same assumption every free GEX tool makes.
- Yahoo is unofficial and occasionally rate-limits. The report just runs again at the next slot.
- Educational research, not financial advice.

## Tweaking

The thresholds and weights live in `build_signals()` in `scripts/build_report.py`. Each `sig(...)` call has its weight as the 4th argument. Market setup (which chains and futures) is in `MARKETS` at the top.
