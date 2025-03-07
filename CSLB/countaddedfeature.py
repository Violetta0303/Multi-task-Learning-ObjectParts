import json

def count_added_features(cached_file_path, log_file_path):
    with open(cached_file_path, 'r') as file:
        cached_data = json.load(file)

    # Count 'True' responses in cached data
    cached_count = sum(1 for response in cached_data.values() if isinstance(response, dict) and response.get('result') == True)

    # Count direct GPT queries from the log file
    gpt_query_count = 0
    with open(log_file_path, 'r') as file:
        for line in file:
            if "Queried GPT for" in line and "Result is Yes" in line:
                gpt_query_count += 1

    # Total count is the sum of cached count and direct GPT query count
    total_added_features = cached_count + gpt_query_count
    return total_added_features

# Paths to the files
cached_file_path = 'cached_responses.json'
log_file_path = 'feature_addition_log.txt'

# Calculate the total number of added features
total_added_features = count_added_features(cached_file_path, log_file_path)
print(f"Total number of features added: {total_added_features}")

