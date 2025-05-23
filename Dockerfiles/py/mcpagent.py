import asyncio
import logging
import os
import sys
import time
import uuid

from mcp_agent.app import MCPApp
from mcp_agent.config import (
    Settings,
    LoggerSettings,
    MCPSettings,
    MCPServerSettings,
    OpenAISettings,
    AnthropicSettings,
)
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.llm_selector import ModelPreferences
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="codex")


async def codex(user_request):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        codex = Agent(
            name="codex",
            instruction="""You are an agent with access to the docker filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls.
            Please do not unnecessarily remove any comments or code.
            Generate the code with clear comments explaining the logic.""",
            server_names=["code-operator", "filesystem"],
        )

        async with codex:
            logger.info("codex: Connected to server, calling list_tools...")
            result = await codex.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await codex.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate_str(
                message=user_request,
            )
            logger.info(f"modified code: {result}")


if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) < 4:
        print("Usage: python swe.py <task_description> <target_file_rel_path> <is_github_repo>")
        sys.exit(1)
    user_request = sys.argv[1] + sys.argv[2] + sys.argv[3]
    asyncio.run(codex(user_request))
    end = time.time()
    t = end - start
    task_id = str(uuid.uuid4())
    logging.log(f"TaskId :{task_id} Total run time: {t:.2f}s")
