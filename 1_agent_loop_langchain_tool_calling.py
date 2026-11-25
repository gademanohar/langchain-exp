from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog. Returns the price as a float. If the product is not found, returns -1.0.
    """
    # Simulated product prices
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "monitor": 299.99,
        "keyboard": 49.99
    }
    return prices.get(product, 0)  # Return 0 if product not found

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price and return the discounted price.
    """
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15, "platinum": 20}

    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="langchain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
        f"ollama:{MODEL}",
        temperature=0,
        reasoning=False,
        num_predict=128,
    )
    llm_with_tools = llm.bind_tools(tools)

    print(f"Running agent with question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant. "
            "You have access to product catalog tool"
            "and a discount tool \n\n"
            "Strict Rules: - you must follow these exactly \n"
            "1. Never guess or assume the product price or discount.\n"
            "you must follow the get_product_price to get the price"
            "2. only call the apply_discount AFTER you have received the price from get_product_price\n"
            "do not pass a made up number.\n"
            "3. Never calculate the discount yourself using math, always use the apply_discount tool to get the discounted price.\n"
            "4. If the user does not specify a discount tier,"
            "ask them which tier to use - do not assume one.\n"
        )),
        HumanMessage(content=question)
    ]
    
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration {iteration}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("No tool calls made. Final response:")
            print(ai_message.content)
            return ai_message.content

        #process one first tool call at a time and force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call: {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))


if __name__ == "__main__":
    question = "What is the price of a laptop with a gold discount?"
    print("Hello Langchain Agent! Let's find out the price of a laptop with a gold discount. (.bind_tools)!")
    print()
    result = run_agent(question)