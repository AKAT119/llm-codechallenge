import json
import openai
import warnings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_KEY
from nltk.translate.bleu_score import sentence_bleu

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define the static Pyomo model template
pyomo_template = """
from pyomo.environ import *

model = ConcreteModel()
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)
model.obj = Objective(expr=2 * model.x + 3 * model.y, sense=maximize)

# Define constraints
model.constraint1 = Constraint(expr=model.x + 2 * model.y <= 8)
model.constraint2 = Constraint(expr=3 * model.x + 2 * model.y <= 12)

{new_constraint}

# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model)
"""

# Define the LangChain prompt template
prompt_template = PromptTemplate(
    input_variables=["template", "user_input"],
    template="Given the following Pyomo model template:\n\n{template}\n\nAdd the following constraint to the model:\n\n{user_input}\n\nand return the complete modified Pyomo code including the new constraint.",
)

# Initialize the OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create the LangChain
chain = LLMChain(prompt=prompt_template, llm=llm)

# Function to interact with the LangChain and generate the final Pyomo code
def interact_with_chain(template, user_input):
    response = chain.invoke({"template": template, "user_input": user_input})
    return response["text"]

# Function to evaluate the model using BLEU score
def evaluate_model(dataset_path):
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
    
    bleu_scores = []

    for data in dataset:
        user_input = data['prompt']
        expected_code = data['code']
        
        generated_code = interact_with_chain(pyomo_template, user_input)
        
        # Compute BLEU score
        reference = [expected_code.split()]
        candidate = generated_code.split()
        bleu_score = sentence_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"Average BLEU Score: {avg_bleu:.2f}")

if __name__ == "__main__":
    dataset_path = "dataset/synthetic_pyomo_data.json"
    evaluate_model(dataset_path)
