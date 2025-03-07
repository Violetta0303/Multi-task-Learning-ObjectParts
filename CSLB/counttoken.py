from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example concept and feature for calculation
example_concept = "apple"  # Example concept (change as needed)
example_feature = "is red"  # Example feature (change as needed)

# Static part of the prompt
static_prompt = """In psycholinguistics, "property norms" refer to sets of typical features or properties that people associate with various object concepts. These features or properties can be thought of as descriptive characteristics of those concepts.

For example, for the concept "apple" people might list properties like "is red," "tastes good", "is a fruit," and "is round". I would like to verify whether a property is true of a concept. For example, "tastes good" and "is red" are properties of the concept "apple", but "is round" and "has teeth" are not properties of the concept "toaster".

Property norms of this type are typically collected from people, in a task where people list the properties that are true of a given concept. You are an advanced AI that is capable of emulating human responses in the property norm task.

Is "{feature}" a property of the concept "{concept}"? Answer based on real-world truth. You can deliberate first on whether the property is true of "{concept}", but then answer finally "yes" or "no". Either "yes" or "no" can potentially be correct; in your deliberations avoid a bias to respond "yes". If there is any ambiguity, or if you are in doubt, please respond "no". You need to consider whether a human would be likely to list this feature as a generally true property of "{concept}". The last text of your response should be either "Final answer: Yes" or "Final answer: No"."""

# Combining static and dynamic parts of the prompt
complete_prompt = static_prompt.format(feature=example_feature, concept=example_concept)

# Tokenize and calculate the number of tokens
num_tokens = len(tokenizer.encode(complete_prompt))

# Estimate the average response length (change as needed based on actual responses)
average_response_length = 20  # An estimated average length of GPT's response

# Total tokens per query (prompt + response)
total_tokens_per_query = num_tokens + average_response_length

print("Number of tokens for the prompt:", num_tokens)
print("Total tokens per query (including response):", total_tokens_per_query)
