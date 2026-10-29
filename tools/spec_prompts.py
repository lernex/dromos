"""Builds the speculative-decoding benchmark prompts: realistic chat requests, rendered with the model's
chat template (direct mode, no thinking) and tokenized, as {name: token ids} for `dromos spec`.

    $PY tools/spec_prompts.py MODEL_DIR OUT_JSON

The five cover how much an answer repeats its context, which is what prompt lookup feeds on:
editing code (the answer is mostly the input), extracting from a pasted document, summarizing it,
a multi-turn chat that revises an earlier answer, and open-ended prose (nothing to copy). Every text
below is written for this benchmark, or is this repository's own documentation.
"""

import json
import pathlib
import sys

from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent.parent

INVENTORY_PY = '''import json
import time
from collections import OrderedDict


class Item:
    def __init__(self, sku, name, price, quantity=0):
        self.sku = sku
        self.name = name
        self.price = price
        self.quantity = quantity

    def total_value(self):
        return self.price * self.quantity

    def to_dict(self):
        return {"sku": self.sku, "name": self.name, "price": self.price, "quantity": self.quantity}


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


class Inventory:
    def __init__(self, cache_size=64):
        self.items = {}
        self.cache = LRUCache(cache_size)
        self.log = []

    def add_item(self, item):
        if item.sku in self.items:
            raise ValueError("duplicate sku " + item.sku)
        self.items[item.sku] = item
        self.log.append(("add", item.sku, time.time()))

    def get_item(self, sku):
        cached = self.cache.get(sku)
        if cached is not None:
            return cached
        item = self.items.get(sku)
        if item is not None:
            self.cache.put(sku, item)
        return item

    def restock(self, sku, amount):
        item = self.get_item(sku)
        if item is None:
            raise KeyError(sku)
        item.quantity += amount
        self.log.append(("restock", sku, time.time()))

    def sell(self, sku, amount):
        item = self.get_item(sku)
        if item is None:
            raise KeyError(sku)
        if item.quantity < amount:
            raise ValueError("not enough stock for " + sku)
        item.quantity -= amount
        self.log.append(("sell", sku, time.time()))
        return item.price * amount

    def low_stock(self, threshold=5):
        return [item for item in self.items.values() if item.quantity < threshold]

    def total_value(self):
        return sum(item.total_value() for item in self.items.values())

    def save(self, path):
        with open(path, "w") as f:
            json.dump([item.to_dict() for item in self.items.values()], f, indent=2)

    def load(self, path):
        with open(path) as f:
            for row in json.load(f):
                self.add_item(Item(row["sku"], row["name"], row["price"], row["quantity"]))


def report(inventory):
    lines = []
    for item in sorted(inventory.items.values(), key=lambda i: i.sku):
        lines.append(f"{item.sku:10s} {item.name:30s} {item.quantity:5d} {item.price:8.2f}")
    lines.append(f"total value: {inventory.total_value():.2f}")
    return "\\n".join(lines)
'''

FIELD_NOTES = """Harbor Valley Water Authority: summary of the 2026 spring operations review

1. Supply. Reservoir storage stood at 71 percent of capacity on April 30, against 64 percent a year
earlier. Inflows from the north fork ran 18 percent above the ten-year average through March, then fell
to 9 percent below it in April after an early snowmelt. The authority expects storage to peak at about
74 percent in mid-May and to fall to 52 percent by the end of September if demand follows last year's
pattern.

2. Demand. Total deliveries in the first quarter were 9.4 million cubic meters, down 3.1 percent from
2025. Residential use per connection fell for the fourth straight year, to 412 liters per day. The
two largest industrial customers, the Linden paper mill and the Corvo bottling plant, together took
1.8 million cubic meters, 19 percent of the total. Corvo has asked for an additional 150,000 cubic
meters a year from 2027 to support a new production line.

3. Infrastructure. The east trunk main replacement reached 6.2 of its planned 11 kilometers. The
contractor reports that the section under Mill Road will finish in July, three weeks late, because
an undocumented gas line was found at the crossing with Station Street. The Ridgeview pumping
station's second motor failed on March 14 and was rebuilt in eleven days; during the outage the
station ran on its single remaining motor and on the standby generator for 31 hours. Non-revenue
water, the share of treated water that is never billed, was 14.8 percent, down from 16.3 percent,
mostly because of the leak detection program in the old town district, which found and repaired 47
leaks.

4. Quality. All 1,284 compliance samples met the drinking water standards. Turbidity at the Westbank
treatment plant exceeded the internal alarm level (not the regulatory limit) twice, on February 2
and February 19, both times after heavy rain; the plant switched to the higher coagulant dose
schedule within an hour each time. A customer complaint about taste in the Elm Park area in March
was traced to a dead-end main that had not been flushed since October; the flushing schedule now
includes all dead-end mains every eight weeks.

5. Finances. Operating revenue for the quarter was 6.9 million, 2 percent under budget, because of the
lower residential demand. Operating costs were 5.7 million, 4 percent under budget, mainly from lower
electricity prices. The capital program spent 3.2 million of its 4.1 million quarterly budget; the
gap is the delayed trunk main section. The board approved a 3.5 percent tariff increase from July 1,
the first since 2023.

6. Risks and actions. The review names three risks for the coming year: a dry summer that would bring
storage under 45 percent by October, which would trigger stage one restrictions; further delay of
the east trunk main beyond the September tie-in window, which would push the final section to the
spring of 2027; and the age of the Ridgeview motors, both installed in 1998. Actions agreed: order a
replacement motor for Ridgeview now rather than in 2027 (lead time 26 weeks); ask the contractor for
a recovery plan for the Mill Road section by May 20; and start talks with Corvo on a supply contract
that ties additional volume to storage levels.
"""

MULTI_TURN = [
    {"role": "user", "content": "I'm planning a three-day trip to Lisbon in October. What should I prioritize?"},
    {"role": "assistant", "content": (
        "October is a good time for Lisbon: warm days, fewer crowds than summer. Here's how I'd use three days.\n\n"
        "Day 1: Alfama and Baixa. Start at the Castelo de Sao Jorge early, before the tour groups, then walk down "
        "through Alfama's lanes to the Se cathedral. In the afternoon, cross Baixa to Praca do Comercio and take the "
        "Elevador de Santa Justa or simply walk up to Chiado for coffee. In the evening, a fado dinner in Alfama.\n\n"
        "Day 2: Belem. Take tram 15 to Belem. See the Jeronimos Monastery (book ahead), the Belem Tower and the "
        "Monument to the Discoveries, and have a pastel de nata at the original bakery. Spend the afternoon at the "
        "MAAT or walking the riverfront back toward the city.\n\n"
        "Day 3: Sintra. Take the train from Rossio (about 40 minutes). Visit Pena Palace in the morning and the "
        "Quinta da Regaleira in the afternoon. Return in time for dinner in Bairro Alto.")},
    {"role": "user", "content": "That's great. I don't love crowds though. Anything I should swap?"},
    {"role": "assistant", "content": (
        "A few swaps keep the plan but avoid the busiest spots.\n\n"
        "In Sintra, go to the Monserrate Palace instead of Pena, or do Pena first thing at opening and skip "
        "Regaleira's well at midday, when the queue is longest. Monserrate's gardens are quiet even in high season.\n\n"
        "In Belem, visit the monastery's church (free, short line) rather than the cloister if the line is long, and "
        "go to the bakery mid-afternoon on a weekday.\n\n"
        "For the evening, skip the big fado houses and look for a small tasca with fado vadio, amateur nights where "
        "locals sing. They're cheaper and less crowded.\n\n"
        "And consider the Gulbenkian Museum as a calm half-day: a superb collection and a garden that is never "
        "busy.")},
    {"role": "user", "content": (
        "Perfect. Please rewrite your first answer with those swaps folded in, as one clean day-by-day plan I can "
        "print. Keep everything else the same.")},
]


def chat(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    docs = (HERE / "docs" / "ARCHITECTURE.md").read_text() + "\n\n" + (HERE / "docs" / "PLAN.md").read_text()
    prompts = {
        "code_edit": chat(tok, [{"role": "user", "content": (
            "Here is a Python module:\n\n```python\n" + INVENTORY_PY + "```\n\nAdd type hints to every function and "
            "method, rename get_item to fetch_item everywhere, and make sell() refuse a non-positive amount with a "
            "ValueError. Return the complete updated file, nothing else.")}]),
        "doc_extract": chat(tok, [{"role": "user", "content": (
            FIELD_NOTES + "\n\nFrom the review above, list every number it reports, each with the sentence it comes "
            "from quoted exactly.")}]),
        "doc_summary": chat(tok, [{"role": "user", "content": (
            "Summarize this engineering document for a new team member in about 300 words, keeping the key numbers."
            "\n\n" + docs)}]),
        "multi_turn": chat(tok, MULTI_TURN),
        "prose": chat(tok, [{"role": "user", "content": (
            "Write an original short story, about 600 words, about a lighthouse keeper on a remote island who "
            "receives a letter addressed to someone who died forty years ago.")}]),
    }
    prompts = {k: [int(t) for t in v] for k, v in prompts.items()}
    pathlib.Path(out).write_text(json.dumps(prompts))
    for k, v in prompts.items():
        print(f"{k:12s} {len(v):5d} tokens")


if __name__ == "__main__":
    main()
