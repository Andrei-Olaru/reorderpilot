# Demo script (4-5 minutes)

**Goal:** show that ReorderPilot (1) tells a rep whom to call, (2) explains a briefing with real numbers from tools,
(3) **learns from feedback**: a declined product is not recommended again, and (4) abstains when it doesn't know.

The customer and product below come from the memory evaluation run (`eval/results/memory_eval.json`):
customer **16121**, top product **85123A WHITE HANGING HEART T-LIGHT HOLDER**. Check them during the rehearsal;
if the top product differs on your machine, use the one the briefing shows.

## Before the demo (5 minutes, off camera)

1. Start a clean memory, so the decline is not already recorded from an earlier run (same simulated date every run):
   rename `data/memory.db` to `data/memory.backup.db`. It is recreated automatically.
2. Start the system: `docker compose up` (or `uvicorn ...` + `streamlit run ...`, see README).
3. Open <http://localhost:8501>. The sidebar should say **API ok · simulated today = 2011-11-01**.
4. Rehearse steps 2-4 once. LLM responses vary a little; know what your answer looks like.
   Then click **New chat session** and delete the rehearsal `data/memory.db` (your original is safe in
   `memory.backup.db`), so the demo starts clean.
5. Check the LLM quota (Groq free tier ≈ 8K tokens/min). Wait 1 minute after the rehearsal.
6. Have `README.md` (section 5, results) and `docs/architecture.md` open in another tab.

## Script

| Time | Click / type | Say |
|---|---|---|
| **0:00-0:30** | Show the UI, sidebar visible. | "A B2B sales rep has hundreds of wholesale customers and no idea whom to call or what to offer. ReorderPilot predicts which products each customer will re-order in the next 30 days, explains why, and remembers what the customer told us. The data is a real UK wholesaler from 2011, so the system lives on a simulated today: 1 November 2011." |
| **0:30-1:15** | Chat: `Which customers should I call today?` | "The agent doesn't answer from its own knowledge. On the right you see the tool calls: `list_customers_due` ranks customers by expected re-order value: probability times usual price times usual quantity." Point at the tool-call panel. |
| **1:15-2:15** | Chat: `Brief me on customer 16121` | "Before recommending anything it must call `predict_reorder` and `read_memory`. That's a prompt rule, and the panel shows it happened. Each product has a calibrated probability: 40% means it happens about 4 times out of 10. The reason comes from the order history, e.g. 'bought 6 times, usually every 21 days'. Every product here came from a tool result; our eval checks that automatically." Point at the top product (T-light holder). |
| **2:15-3:30** | **Memory scenario.** Chat: `Customer 16121 said the white hanging heart T-light holders are still overstocked.` | "Now the rep logs the call outcome in plain English. The agent finds the stock code with `find_product`, then calls `write_memory`: outcome *declined*, reason *overstocked*." Point at the `write_memory` args in the panel. "The rule 'overstocked means hide it for 30 days' is in code, not left to the LLM." |
| | Open the **Customer memory** tab (customer 16121). | "Here is the stored event, valid until 1 December. Below are the model's current predictions: the T-light holder is still in the list, but flagged *suppressed* with the reason. We flag instead of deleting, so you can always see why something wasn't offered." |
| | Back to **Chat**: `Brief me on customer 16121 again.` | "The new briefing doesn't recommend the T-light holder. It mentions that it was declined as overstocked and offers the next products instead. In our memory evaluation, the declined item was repeated TODO(numbers)% of the time with memory on, vs TODO(numbers)% with memory off." |
| **3:30-4:00** | Chat: `Brief me on customer <COLD_START_ID>` (pick one with < 3 orders during the rehearsal). | "This customer has fewer than 3 orders. The model abstains, and the agent says 'not enough history' instead of guessing. A wrong confident offer costs the rep's trust." |
| **4:00-4:45** | Switch to README section 5. | "Evidence: on the last three months, held out, the model beats the best simple rule on precision@5 (TODO(numbers) vs TODO(numbers)) and PR-AUC. Limitations: one retailer, the test months are the Christmas season, and the agent evals ran on small samples because of the free LLM quota." |
| **4:45-5:00** | Show the architecture diagram. | "Streamlit talks to FastAPI. A hand-written agent loop calls the LLM and six tools. The memory is SQLite with deterministic rules. Questions?" |

## If something goes wrong

- **LLM rate limit / slow answer:** say "free-tier quota, the client waits and retries, which is designed in", or
  skip the chat step and log the decline with the **Log feedback** form in the memory tab
  (outcome `declined`, stock code `85123A`, reason `overstocked`). The suppression appears the same way, because the
  form and the agent share the same policy code.
- **"Who should I call today?" refused as out-of-scope** (seen once in eval D): use the **Load customers due**
  button in the sidebar, and mention it as a known issue that prompt v2 fixes.
- **Agent writes `asked_alternative` instead of `declined`:** show it honestly: the tool panel makes the mistake
  visible. Fix it with the form. This is the known failure from eval C.

## After the demo

Restore your memory: delete the demo `data/memory.db` and rename `memory.backup.db` back.
