import openai
import json
import os

openai.api_key = 'YOUR-API-KEY'

# Load existing namemap
if os.path.exists('namemap.json'):
    with open('namemap.json', 'r') as file:
        namemap = json.load(file)
else:
    namemap = {}

# Load processed status and log (if they exist)
if os.path.exists('processed_map.json'):
    with open('processed_map.json', 'r') as file:
        processed_terms = json.load(file)
else:
    processed_terms = {}

if os.path.exists('map_log.json'):
    with open('map_log.json', 'r') as file:
        query_log = json.load(file)
else:
    query_log = []

def query_gpt(cslb_term, things_term):
    prompt = f"Consider if the term '{cslb_term}' is an equivalent or closely related to '{things_term}' in formal English, aiming for real-world accuracy. If '{cslb_term}' and '{things_term}' are exactly the same, you should immediately conclude with 'Final answer: Yes'. For all other cases, deliberation on nuances is welcome, but the conclusion must be definitive, either 'yes' or 'no'. The evaluation should mirror human judgment, being precise without being unnecessarily strict. For instance, 'coffee machine' can be considered synonymous with 'coffee maker', despite slight differences. Conclude with 'Final answer: Yes' or 'Final answer: No'."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

updated_namemap = {}

# Adding counter and total length for progress tracking
total_terms = len(namemap)
processed_count = 0

for cslb_term, things_term in namemap.items():
    processed_count += 1  # Increment the counter
    if cslb_term not in processed_terms:
        print(f"Processing {processed_count}/{total_terms}: {cslb_term}")  # Print current progress
        verification_result = query_gpt(cslb_term, things_term)
        print(f"Verification result for '{cslb_term}' and '{things_term}': {verification_result}\n")  # Print the GPT's response to console
        query_log.append({'cslb_term': cslb_term, 'things_term': things_term, 'verification': verification_result})
        processed_terms[cslb_term] = "yes" in verification_result.lower()
        # Update and save the updated namemap
        if processed_terms[cslb_term]:
            updated_namemap[cslb_term] = things_term
        with open('updated_namemap.json', 'w') as file:
            json.dump(updated_namemap, file, indent=4)
        # Save log and processed status
        with open('map_log.json', 'w') as file:
            json.dump(query_log, file, indent=4)
        with open('processed_map.json', 'w') as file:
            json.dump(processed_terms, file, indent=4)

# Optionally, save the final updated namemap again (for redundancy)
with open('updated_namemap.json', 'w') as file:
    json.dump(updated_namemap, file, indent=4)