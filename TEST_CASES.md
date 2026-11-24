# Veritas Desk: Test Cases Guide (TC-01 to TC-18)

Use this guide to copy-paste test headlines and URLs directly into the **Veritas Desk** interface (`http://127.0.0.1:5000/`) to verify all conditions.

> [!NOTE]
> Veritas Desk operates with Google Search grounding and deterministic guardrails. When `GEMINI_API_KEY` is configured in `.env`, factual claims are dynamically corroborated via real-time search evidence. If `GEMINI_API_KEY` is not set, factual claims gracefully return `UNAVAILABLE` without falling back to misleading style guesses, while non-claims (questions, predictions, opinions, satire) are immediately resolved by the transparent Input Gate.

---

## 1. Quick Master Table

| Test ID | Condition Tested | Test Headline (Text to Paste) | Source URL (Optional) | Expected Verdict | Expected Confidence |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **TC-01** | Genuine News (Economy) | `Central bank keeps interest rates unchanged citing stable inflation and steady economic growth` | *(Leave Empty)* | **SUPPORTED** | **High** |
| **TC-02** | Genuine News (Space) | `India launches Chandrayaan-3 lunar exploration mission from Sriharikota spaceport` | *(Leave Empty)* | **SUPPORTED** | **High** |
| **TC-03** | Genuine News + Verified URL | `Supreme Court stays Allahabad HC order on survey of Shahi Idgah complex` | `https://indianexpress.com/article/india/supreme-court-stays-allahabad-hc-order-survey-shahi-idgah/` | **SUPPORTED** | **High** |
| **TC-04** | Genuine News (Judiciary) | `Supreme Court delivers verdict declaring electoral bonds scheme unconstitutional` | `https://thehindu.com` | **SUPPORTED** | **High** |
| **TC-05** | Fake News (Medical Clickbait) | `Scientists confirm hot water cures all cancers, doctors hate this secret!` | *(Leave Empty)* | **REFUTED** | **High** |
| **TC-06** | Fake News (Conspiracy Hoax) | `NASA officially admitted that the Moon is made of green cheddar cheese` | *(Leave Empty)* | **REFUTED** | **High** |
| **TC-07** | Input Gate: Future Prediction | `Share before it gets deleted: banks will freeze all accounts tomorrow withdraw cash now` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-08** | Input Gate: Future Prediction | `The regional weather forecast expects heavy rainfall next weekend` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-09** | Input Gate: Question | `Will it rain today in Hyderabad?` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-10** | Input Gate: Question | `Is the Reserve Bank of India planning to cut interest rates next month?` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-11** | Input Gate: Editorial / Opinion | `Editorial: The newly proposed education policy is a complete failure` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-12** | Satire URL Penalty (The Onion) | `Nation's Optimists Conclude Bad Things Never Happen` | `https://theonion.com/article` | **NOT_A_CLAIM** | **High (Satire Guardrail)** |
| **TC-13** | Satire URL Penalty (Babylon Bee) | `Government Efficiently Solves Major Problem Ahead of Schedule` | `https://babylonbee.com/article` | **NOT_A_CLAIM** | **High (Satire Guardrail)** |
| **TC-14** | Disreputable URL Guardrail | `Secret chemicals detected in water supply` | `https://infowars.com/article` | **NOT_ENOUGH_EVIDENCE** | **Low (Disreputable)** |
| **TC-15** | Time-Sensitive Outdated Guard | `Heavy rain hit Mumbai today` *(paired with search evidence older than 2 days)* | *(Leave Empty)* | **NOT_ENOUGH_EVIDENCE** | **Low (Time Guardrail)** |
| **TC-16** | Uncorroborated Rumor | `Ancient alien ruins discovered beneath Antarctica ice sheet yesterday` | *(Leave Empty)* | **NOT_ENOUGH_EVIDENCE** | **Low (No Evidence)** |
| **TC-17** | Input Gate: Too Short (< 5 words)| `Short breaking news` | *(Leave Empty)* | **NOT_A_CLAIM** | **High (Gate)** |
| **TC-18** | Boundary Limit (> 4000 Chars) | `Central bank announces interest rate policy update ` *(repeated 100+ times)* | *(Leave Empty)* | **HTTP 400** | **"exceeds max length"** |

---

## 2. Copy-and-Paste Test Blocks

---

### TC-01: Genuine News (Economy)
* **Headline:**
```text
Central bank keeps interest rates unchanged citing stable inflation and steady economic growth
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **SUPPORTED (High confidence)**

---

### TC-02: Genuine News (Space)
* **Headline:**
```text
India launches Chandrayaan-3 lunar exploration mission from Sriharikota spaceport
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **SUPPORTED (High confidence)**

---

### TC-03: Genuine News + Verified Indian Express URL
* **Headline:**
```text
Supreme Court stays Allahabad HC order on survey of Shahi Idgah complex
```
* **Source URL:**
```text
https://indianexpress.com/article/india/supreme-court-stays-allahabad-hc-order-survey-shahi-idgah/
```
* **Expected Verdict:** **SUPPORTED (High confidence)**
* **Publisher:** The Indian Express (Recognized as `trusted`)

---

### TC-04: Genuine News (Judiciary)
* **Headline:**
```text
Supreme Court delivers verdict declaring electoral bonds scheme unconstitutional
```
* **Source URL:**
```text
https://thehindu.com
```
* **Expected Verdict:** **SUPPORTED (High confidence)**

---

### TC-05: Fake News (Medical Clickbait Scam)
* **Headline:**
```text
Scientists confirm hot water cures all cancers, doctors hate this secret!
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **REFUTED (High confidence)**

---

### TC-06: Fake News (Conspiracy Hoax)
* **Headline:**
```text
NASA officially admitted that the Moon is made of green cheddar cheese
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **REFUTED (High confidence)**

---

### TC-07: Input Gate: Future Prediction
* **Headline:**
```text
Share before it gets deleted: banks will freeze all accounts tomorrow withdraw cash now
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `prediction` (Bypasses external LLM in < 10ms)

---

### TC-08: Input Gate: Weather Forecast
* **Headline:**
```text
The regional weather forecast expects heavy rainfall next weekend
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `prediction`

---

### TC-09: Input Gate: Question (Interrogative)
* **Headline:**
```text
Will it rain today in Hyderabad?
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `question` (Bypasses external LLM in < 10ms with rephrasing advice)

---

### TC-10: Input Gate: Question (Auxiliary Verb)
* **Headline:**
```text
Is the Reserve Bank of India planning to cut interest rates next month?
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `question`

---

### TC-11: Input Gate: Editorial / Opinion
* **Headline:**
```text
Editorial: The newly proposed education policy is a complete failure
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `opinion`

---

### TC-12: Satire Domain Penalty (The Onion)
* **Headline:**
```text
Nation's Optimists Conclude Bad Things Never Happen
```
* **Source URL:**
```text
https://theonion.com/article
```
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Guardrail Note:** `Source domain 'theonion.com' is identified as a satire publication; forced verdict to NOT_A_CLAIM.`

---

### TC-13: Satire Domain Penalty (Babylon Bee)
* **Headline:**
```text
Government Efficiently Solves Major Problem Ahead of Schedule
```
* **Source URL:**
```text
https://babylonbee.com/article
```
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Guardrail Note:** `Source domain 'babylonbee.com' is identified as a satire publication; forced verdict to NOT_A_CLAIM.`

---

### TC-14: Disreputable Domain Guardrail
* **Headline:**
```text
Secret chemicals detected in water supply
```
* **Source URL:**
```text
https://infowars.com/article
```
* **Expected Verdict:** **NOT_ENOUGH_EVIDENCE (Low confidence)**
* **Guardrail Note:** `Source domain 'infowars.com' is a flagged disreputable source; forced verdict to NOT_ENOUGH_EVIDENCE.`

---

### TC-15: Time-Sensitive Outdated Evidence Guardrail
* **Headline:**
```text
Heavy rain hit Mumbai today
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_ENOUGH_EVIDENCE (Low confidence)** *(If evidence articles are older than 2 days)*
* **Guardrail Note:** `Claim is time-sensitive, but evidence dates are older than 2 days or unknown.`

---

### TC-16: Uncorroborated Speculative Assertion
* **Headline:**
```text
Ancient alien ruins discovered beneath Antarctica ice sheet yesterday
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_ENOUGH_EVIDENCE (Low confidence)**

---

### TC-17: Input Gate: Sub-Minimum Word Length Check (< 5 words)
* **Headline:**
```text
Short breaking news
```
* **Source URL:** *(Leave empty)*
* **Expected Verdict:** **NOT_A_CLAIM (High confidence)**
* **Input Type:** `too_short`

---

### TC-18: Boundary: Buffer Overflow Check (> 4000 Characters)
* **Headline:**
```text
Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update. Central bank announces interest rate policy update.
```
* **Source URL:** *(Leave empty)*
* **Expected:** **HTTP 400 Bad Request** (`"Text exceeds maximum length of 4,000 characters."`)

---

## 3. Daily Activity Category Verification (Sports, Weather, Entertainment, Politics, Incidents)

These test cases demonstrate that uncorroborated daily claims with zero trusted news reports are strictly capped at **Unverified (50%)** and marked **NOT_ENOUGH_EVIDENCE**, preventing well-written fake news from masquerading as real.

---

### TC-19: Weather (Uncorroborated / Fabricated Weather Event)
* **Headline:**
```text
hyderabad noticed heavy rainfall on 24th november 2026
```
* **Source URL:** *(Leave empty)*
* **Expected Category:** `Weather & Meteorology (🌦️)`
* **Expected Verdict:** **Unverified (NOT_ENOUGH_EVIDENCE)**
* **Expected Score:** **50%**
* **Expected Confidence:** **Low**
* **Notice:** `Uncorroborated Weather Notice: Zero corroborating news reports found from trusted publishers.`

---

### TC-20: Sports (Uncorroborated / Fabricated Match Result)
* **Headline:**
```text
India won the football world cup final against Brazil yesterday
```
* **Source URL:** *(Leave empty)*
* **Expected Category:** `Sports & Athletics (🏏)`
* **Expected Verdict:** **Unverified (NOT_ENOUGH_EVIDENCE)**
* **Expected Score:** **50%**
* **Expected Confidence:** **Low**
* **Notice:** `Uncorroborated Sports Notice: Zero corroborating sports match reports found.`

---

### TC-21: Entertainment (Uncorroborated Celebrity Rumor)
* **Headline:**
```text
Shah Rukh Khan announced retirement from acting today
```
* **Source URL:** *(Leave empty)*
* **Expected Category:** `Entertainment & Culture (🎬)`
* **Expected Verdict:** **Unverified (NOT_ENOUGH_EVIDENCE)**
* **Expected Score:** **50%**
* **Expected Confidence:** **Low**
* **Notice:** `Uncorroborated Entertainment Notice: No official statement or entertainment media corroboration.`

---

### TC-22: Politics (Uncorroborated Policy Fabrication)
* **Headline:**
```text
Prime Minister declared national holiday for the whole month
```
* **Source URL:** *(Leave empty)*
* **Expected Category:** `Politics & Governance (🏛️)`
* **Expected Verdict:** **Unverified (NOT_ENOUGH_EVIDENCE)**
* **Expected Score:** **50%**
* **Expected Confidence:** **Low**
* **Notice:** `Uncorroborated Politics Notice: No official government gazette or press confirmation.`

---

### TC-23: Disaster / Incident (Uncorroborated Earthquake Rumor)
* **Headline:**
```text
Earthquake hits California today with magnitude 8.5
```
* **Source URL:** *(Leave empty)*
* **Expected Category:** `Incidents & Disasters (🚨)`
* **Expected Verdict:** **Unverified (NOT_ENOUGH_EVIDENCE)**
* **Expected Score:** **50%**
* **Expected Confidence:** **Low**
* **Notice:** `Uncorroborated Incident Notice: No USGS or news reports confirm this event.`
