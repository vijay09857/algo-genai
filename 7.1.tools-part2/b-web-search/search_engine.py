import asyncio
from dataclasses import dataclass
import json
import os
import re
import sys
from pydantic import BaseModel
from ddgs import DDGS
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from dotenv import load_dotenv
import logfire

load_dotenv(override=True)
logfire.configure()
logfire.instrument_pydantic_ai()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Add it to your environment or .env file before running."
    )


@dataclass
class SearchDeps:
    max_results: int = 5


search_agent = Agent(
    model=GroqModel(
        "llama-3.3-70b-versatile",
        provider=GroqProvider(api_key=groq_api_key),
    ),
    # 'github:openai/gpt-4o',
    # "ollama:llama3-groq-tool-use:8b",
    # "ollama:qwen3.5:9b",
    deps_type=SearchDeps,
    instructions="You are a helpful assistant. Use the tools provided to find requested infomration."
)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def run_web_search(query: str, max_results: int) -> list[SearchResult]:
    results = []
    with DDGS() as ddgs:
        search_results = ddgs.text(query, max_results=max_results)
        for r in search_results:
            results.append(SearchResult(
                title=r.get('title', ''),
                url=r.get('href', ''),
                snippet=r.get('body', '')
            ))
    return results


@search_agent.tool
def web_search(ctx: RunContext[SearchDeps], query: str) -> list[SearchResult]:
    """
    Perform a web search.

    Args:
        query: The search query.
        max_results: The number of results to return (default 5).
    """
    return run_web_search(query=query, max_results=ctx.deps.max_results)


async def main(user_prompt):
    search_deps = SearchDeps()
    response = await search_agent.run(user_prompt, deps=search_deps)
    output = str(response.output)
    function_match = re.search(
        r"<function=([a-zA-Z0-9_.-]+)\s*(\{.*?\})\s*</function>",
        output,
        flags=re.DOTALL,
    )

    if function_match and function_match.group(1) in {"web_search", "web.search"}:
        print("Model returned a tool call as text; running fallback search...")
        function_payload = function_match.group(2)
        function_args = json.loads(function_payload)
        query = function_args.get("query", user_prompt)
        results = run_web_search(query=query, max_results=search_deps.max_results)

        print(f"Top {len(results)} results for: {query}")
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. {result.title}")
            print(f"   {result.url}")
            if result.snippet:
                print(f"   {result.snippet}")
        return

    print(output)


if __name__ == "__main__":
    user_prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "search web for the top 3 news stories in GenerativeAI today?"
    )
    asyncio.run(main(user_prompt))

# "search web for the top 3 news stories in GenerativeAI today?"
# "search web for the latest updates on israel, usa and iran conflict"