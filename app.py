
import os
import ast
import operator
import streamlit as st

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# 1. Load API key from .env
load_dotenv()

MODEL_NAME = "openai:gpt-4o-mini"

SYSTEM_PROMPT = """
You are a Personal Finance AI Assistant.
Use the appropriate Python tool for every financial calculation.
Use the EMI tool for loans, SIP tool for investment projections,
budget tool for monthly budgets, and calculator for arithmetic.
Explain assumptions and show financial results in Indian rupees.
Investment returns are estimates and are not guaranteed.
Ask for missing inputs instead of inventing them.
"""


# 2. Safe arithmetic calculator

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(node):
    if isinstance(node, ast.Expression):
        return evaluate_expression(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.UnaryOp):
        fn = OPERATORS.get(type(node.op))
        if fn is None:
            raise ValueError("Unsupported operator")
        return fn(evaluate_expression(node.operand))

    if isinstance(node, ast.BinOp):
        fn = OPERATORS.get(type(node.op))
        if fn is None:
            raise ValueError("Unsupported operator")

        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > 10 or abs(left) > 1_000_000:
                raise ValueError("Power operation too large")

        result = fn(left, right)

        if not isinstance(result, (int, float)) or abs(result) > 1e100:
            raise ValueError("Result out of range")

        return result

    raise ValueError("Use only numbers and basic arithmetic operators")


@tool
def calculator_tool(expression: str) -> str:
    """Calculate arithmetic, e.g. '85000 * 0.20'."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate_expression(tree)
        return f"Result: {result:,.2f}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# 3. EMI calculator

@tool
def emi_calculator(
    principal: float,
    annual_interest_rate: float,
    tenure_years: int,
) -> str:
    """Calculate loan EMI, total repayment and interest.
    principal is in rupees; annual_interest_rate is a percentage;
    tenure_years is the duration of the loan in years.
    """

    if principal <= 0 or tenure_years <= 0:
        return "Error: loan amount and tenure must be positive."

    if annual_interest_rate < 0 or tenure_years > 100:
        return "Error: invalid interest rate or tenure."

    months = tenure_years * 12
    rate = annual_interest_rate / 1200

    if rate == 0:
        emi = principal / months
    else:
        factor = (1 + rate) ** months
        emi = principal * rate * factor / (factor - 1)

    total = emi * months

    return (
        f"Monthly EMI: ₹{emi:,.2f}\n"
        f"Total repayment: ₹{total:,.2f}\n"
        f"Total interest: ₹{total - principal:,.2f}"
    )


# 4. SIP calculator

@tool
def sip_calculator(
    monthly_investment: float,
    annual_return_rate: float,
    tenure_years: int,
    target_amount: float = 0,
) -> str:
    """Calculate future value of monthly investments or the monthly
    SIP required for a target. Rates are annual percentages.
    For a target calculation, set monthly_investment to 0
    and provide target_amount.
    """

    if monthly_investment < 0 or target_amount < 0:
        return "Error: amounts cannot be negative."

    if annual_return_rate < 0 or not 0 < tenure_years <= 100:
        return "Error: invalid rate or tenure."

    if monthly_investment == 0 and target_amount == 0:
        return "Provide a monthly investment or target amount."

    months = tenure_years * 12
    rate = annual_return_rate / 1200

    factor = (
        months if rate == 0
        else ((1 + rate) ** months - 1) / rate
    )

    if monthly_investment == 0:
        required = target_amount / factor
        return (
            f"Target: ₹{target_amount:,.2f}\n"
            f"Required monthly SIP: ₹{required:,.2f}"
        )

    invested = monthly_investment * months
    future_value = monthly_investment * factor

    result = (
        f"Total invested: ₹{invested:,.2f}\n"
        f"Estimated future value: ₹{future_value:,.2f}\n"
        f"Estimated gain: ₹{future_value - invested:,.2f}"
    )

    if target_amount > 0:
        result += (
            f"\nTarget reached: {future_value >= target_amount}"
        )

    return result + "\nReturns are illustrative, not guaranteed."


# 5. Budget planner

@tool
def budget_planner(monthly_income: float) -> str:
    """Allocate monthly take-home income using the 50-30-20 rule."""

    if monthly_income <= 0:
        return "Error: income must be positive."

    return (
        f"Monthly income: ₹{monthly_income:,.2f}\n"
        f"Needs (50%): ₹{monthly_income * 0.50:,.2f}\n"
        f"Wants (30%): ₹{monthly_income * 0.30:,.2f}\n"
        f"Savings (20%): ₹{monthly_income * 0.20:,.2f}"
    )


# 6. Create LangChain agent

TOOLS = [
    calculator_tool,
    emi_calculator,
    sip_calculator,
    budget_planner,
]


@st.cache_resource
def get_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "API key missing. Add OPENAI_API_KEY to your .env file."
        )

    return create_agent(
        model=MODEL_NAME,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# 7. Streamlit user interface

st.set_page_config(
    page_title="Personal Finance AI Agent",
    page_icon="💰",
)

st.title("💰 Personal Finance AI Agent")
st.write("Ask questions about loans, SIPs, budgets and calculations.")

st.info(
    "Financial projections are estimates, not guarantees or "
    "personalized investment advice."
)

examples = [
    "Calculate EMI for a ₹10 lakh loan at 8% for 20 years.",
    "Invest ₹7000 monthly for 15 years at 10% return.",
    "Create a budget for income of ₹90000.",
    "What is 20% of ₹85000?",
    "How much will ₹3000 monthly grow to in 5 years at 8%?",
    "How much should I invest monthly to reach ₹20 lakh "
    "in 10 years at 10% return?",
]

with st.sidebar:
    st.header("Example questions")
    for example in examples:
        st.markdown(f"- {example}")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("tools"):
            st.caption("Tools used: " + ", ".join(item["tools"]))

question = st.chat_input("Enter your financial question")

if question:
    st.session_state.history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Selecting the appropriate tool..."):
                response = get_agent().invoke({
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                })

            messages = response["messages"]
            answer = str(messages[-1].content)

            used_tools = []
            for message in messages:
                for call in getattr(message, "tool_calls", []):
                    name = call.get("name")
                    if name and name not in used_tools:
                        used_tools.append(name)

            st.markdown(answer)

            if used_tools:
                st.caption("Tools used: " + ", ".join(used_tools))
            else:
                st.caption("No tool was called for this response.")

            st.session_state.history.append({
                "role": "assistant",
                "content": answer,
                "tools": used_tools,
            })

        except Exception as exc:
            st.error(f"Error: {exc}")
