"""
Initial draft to test changing components of LLMs with capsule network properties in order to optimize metrics like accuracy and perplexity. 
"""

import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Define the population size and number of generations
population_size = 10
num_generations = 10

# Load pre-trained LLM model and tokenizer
llm_model_names = ["gpt2", "gpt2-medium", "gpt2-large", "bert-base-uncased", "roberta-base"]
llm_tokenizers = {model_name: AutoTokenizer.from_pretrained(model_name) for model_name in llm_model_names}

# Define capsule network parameters
capsule_layers = [64, 128, 256]  # Example: Different capsule layer sizes
num_capsules = [4, 8, 16]  # Example: Different numbers of capsules
routing_iterations = [2, 3, 4]  # Example: Different numbers of routing iterations

# Define evaluation metric (combination of perplexity and capsule network performance)
def evaluate_hybrid_model(llm_model, llm_tokenizer, capsule_layer_size, num_capsules, routing_iterations):
    # Dummy evaluation function for illustration
    accuracy = random.uniform(80, 100)  # Simulated accuracy score
    return accuracy

# Evaluation function to compute combined metric and set 'combined_metric' key
def evaluate_and_set_combined_metric(architecture):
    llm_model_name = architecture["llm_model_name"]
    llm_tokenizer = llm_tokenizers[llm_model_name]
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

    # Evaluate hybrid model performance
    accuracy = evaluate_hybrid_model(llm_model, llm_tokenizer,
                                     architecture["capsule_layer_size"],
                                     architecture["num_capsules"],
                                     architecture["routing_iterations"])
    architecture['accuracy'] = accuracy

# Generate initial population of hybrid architectures
def create_population():
    population = []
    for _ in range(population_size):
        # Generate random hybrid architecture parameters
        hybrid_architecture = {
            "llm_model_name": random.choice(llm_model_names),
            "capsule_layer_size": random.choice(capsule_layers),
            "num_capsules": random.choice(num_capsules),
            "routing_iterations": random.choice(routing_iterations)
        }
        population.append(hybrid_architecture)
    return population

# Evolutionary loop
top_model = None
for generation in range(num_generations):
    print(f"Generation {generation + 1}")
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(total=population_size, desc=f'Generation {generation + 1}', unit='model')
    
    # Evaluate each model in the population
    population = create_population()
    while True:
        for architecture in population:
            try:
                evaluate_and_set_combined_metric(architecture)
            except Exception as e:
                print(f"Error evaluating model: {e}")
                architecture['accuracy'] = 0.0  # Set accuracy to 0.0 in case of error
            progress_bar.update(1)
        
        # Filter models with accuracy >= 95%
        filtered_population = [architecture for architecture in population if architecture['accuracy'] >= 95]
        if filtered_population:
            population = filtered_population
            break
        else:
            print("No models with accuracy >= 95%, resetting population.")
            population = create_population()
    
    # Close tqdm progress bar
    progress_bar.close()
    
    # Sort models by accuracy (higher is better)
    population.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print top models
    top_model = population[0]
    print(f"Top Model {generation + 1}: {top_model['llm_model_name']}")
    print(f"Accuracy: {top_model['accuracy']}")
    print(f"Is Hybrid: {'Yes' if top_model['llm_model_name'] in llm_model_names else 'No'}")
    print("Configurations:")
    print(f"  Capsule Layer Size: {top_model['capsule_layer_size']}")
    print(f"  Number of Capsules: {top_model['num_capsules']}")
    print(f"  Routing Iterations: {top_model['routing_iterations']}")
    print()
    
# Final top model details
if top_model:
    print("Final Top Model Details:")
    print(f"Model: {top_model['llm_model_name']}")
    print(f"Accuracy: {top_model['accuracy']}")
    print(f"Is Hybrid: {'Yes' if top_model['llm_model_name'] in llm_model_names else 'No'}")
    print("Configurations:")
    print(f"  Capsule Layer Size: {top_model['capsule_layer_size']}")
    print(f"  Number of Capsules: {top_model['num_capsules']}")
    print(f"  Routing Iterations: {top_model['routing_iterations']}")

    # Instructions to recreate the top model
    print("\nInstructions to recreate the top model:")
    print("# Instantiate the tokenizer")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{top_model['llm_model_name']}')")
    print("# Instantiate the model")
    print(f"model = AutoModelForCausalLM.from_pretrained('{top_model['llm_model_name']}')")
    print("# Model configuration")
    print(f"model.config.capsule_layer_size = {top_model['capsule_layer_size']}")
    print(f"model.config.num_capsules = {top_model['num_capsules']}")
    print(f"model.config.routing_iterations = {top_model['routing_iterations']}")
