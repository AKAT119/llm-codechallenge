import openai
import warnings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_KEY

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Solve the model
solver = SolverFactory('glpk')
result = solver.solve(model)
"""

# Define the LangChain prompt template
prompt_template = PromptTemplate(
    input_variables=["template", "user_input"],
    template = """Given the following Pyomo model template:{template} .Add the following code to the model, user input:{user_input} and return the complete modified Pyomo code. Ensure that the mathematical signs in the given code match exactly what is given in the user input, without any changes. Do not replace '>' with '>=' or vice versa follow user input strictly.""")


# Initialize the OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

# Create the LangChain
chain = LLMChain(prompt=prompt_template, llm=llm)

# Function to interact with the LangChain and generate the final Pyomo code
def interact_with_chain(template, user_input):
    response = chain.invoke({"template": template, "user_input": user_input})
    return response["text"]

if __name__ == "__main__":
    # Prompt the user for input
    user_input = input("Please enter the details you want to add to the code: ")

    # Generate the Pyomo code using LangChain and OpenAI
    pyomo_code = interact_with_chain(pyomo_template, user_input)
    print(pyomo_code)