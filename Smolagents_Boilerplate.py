from smolagents import tool, ToolCallingAgent, CodeAgent
from smolagents.models import InferenceClientModel
from smolagents.models import OpenAIServerModel

# === Load Model ===
# This is the language model (LLM) That acts a brain & enbles them to think.
model = InferenceClientModel("Qwen/Qwen2.5-7B-Instruct")

# === Or === #

# === Use OpenRouter ===

# === Load Model ===

model = OpenAIServerModel(
    model_id="meta-llama/llama-3.3-70b-instruct",  
    api_key=# valid api key,
    api_base="https://openrouter.ai/api/v1/"
)


# === Define Support Tools ===
# These are Python functions that perform specific tasks.
# We Have to mark it with a @tool so that agents can discover and use them.
# Add the Tools based on the requirements.

@tool
def track_order(order_id: str) -> str:
    """
    Track the status of an order.

    Args:
        order_id (str): The ID of the order to track.

    Returns:
        str: A message indicating the current status of the order.
    """
    return f"🔍 Order {order_id} is in transit and will be delivered in 2 days."

@tool
def check_product_availability(product_name: str) -> str:
    """
    Check if a product is currently available.

    Args:
        product_name (str): The name of the product to check.

    Returns:
        str: A message indicating whether the product is in stock.
    """
    return f"✅ {product_name} is currently in stock."

@tool
def get_product_info(product_name: str) -> str:
    """
    Get information about a product.

    Args:
        product_name (str): The name of the product to get information about.

    Returns:
        str: A message containing details about the product.
    """
    return f"📦 {product_name} is made of recycled aluminum and ships with a 1-year warranty."

# === Specialized ToolCalling Agents ===

# Each agent is assigned a specific responsibility and can access only the tools relevant to its role.

order_agent = ToolCallingAgent(
    name="OrderTrackingAgent",
    model=model,
    tools=[track_order],
    description="Handles queries related to tracking the status of customer orders."
)

product_agent = ToolCallingAgent(
    name="ProductAgent",
    model=model,
    tools=[check_product_availability, get_product_info],
    description="Manages queries about product availability and product information."
)

# === Manager Agent with Planning ===
# This is the central manager that coordinates all other agents.
# It receives user queries, understands them, and delegates tasks to the right specialist agent.

manager_agent = CodeAgent(
    name="CustomerServiceManager",
    model=model,
    managed_agents=[
        order_agent,
        product_agent
    ],
    tools=[],  # You can include any globally available tools here
    verbosity_level=2,  
    max_steps=8
)

# === Custom System Prompt for Manager ===
# This instructs the manager agent how to reason through problems using a clear structure.
custom_system_prompt = (
    "You are a customer service manager assistant.\n"
    "Your job is to understand customer issues, determine which support agent to call, and handle the query end-to-end.\n"
    "Follow this format:\n"
    "Thought: Analyze the customer's input.\n"
    "Action: Call a tool or agent with arguments.\n"
    "Observation: Record the response.\n"
    "Repeat Thought/Action/Observation as needed until the issue is resolved."
)

# Append the custom prompt to the agent's default system prompt.
manager_agent.prompt_templates["system_prompt"] = manager_agent.prompt_templates["system_prompt"] + custom_system_prompt

# === Sample Customer Queries ===
# This is the entry point for the program.
# You provide a real-world query, and the manager will coordinate the right agent to solve it.

if __name__ == "__main__":
    query = (
        "Hi, I ordered a phone case last week and still haven’t received it. "
        "Can you check order ID 789123?"
    )

    final_response = manager_agent.run(query)
    print("\nFINAL RESPONSE:\n", final_response)
