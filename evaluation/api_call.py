import asyncio
import random
from itertools import cycle

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


async def get_completion_async(
    client: AsyncOpenAI,
    model_name: str,
    prompt,
    return_reasoning: bool,
    max_retry: int,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    semaphore: asyncio.Semaphore,
):
    """Call one prompt with retry/backoff under a per-client concurrency limit."""
    async with semaphore:
        backoff = 1.5
        is_gpt5 = model_name.startswith("gpt-5")

        for attempt in range(1, max_retry + 1):
            try:
                if is_gpt5:
                    resp = await client.responses.create(
                        model=model_name,
                        input=prompt,
                        max_output_tokens=max_completion_tokens,
                    )

                    output_text = resp.output_text or ""
                    assert len(output_text) > 10

                    if return_reasoning:
                        reasoning_text = ""
                        rc = getattr(resp, "reasoning_content", None)
                        if rc:
                            reasoning_text = "".join(
                                getattr(block, "text", "") for block in rc
                            )
                        return output_text, reasoning_text
                    else:
                        return output_text

                else:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=prompt,
                        temperature=temperature,
                        # top_p=top_p,
                        max_tokens=max_completion_tokens,
                    )

                    content = resp.choices[0].message.content or ""

                    if return_reasoning:
                        return content, None
                    else:
                        return content

            except Exception as e:
                if attempt == max_retry:
                    continue
                print(f"[{model_name}] Error on attempt {attempt}: {e}")
                await asyncio.sleep(backoff * (2 ** (attempt - 1)) + random.random())


async def call_with_index(index: int, **kwargs):
    """Keep async results aligned with their original prompt index."""
    try:
        result = await get_completion_async(**kwargs)
        return index, result
    except Exception:
        return index, None


async def api_call_async(
        model_name: str, 
        user_prompt_list: list[str], 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512) -> list[tuple[str, str] | str | None]:
    """Fan out prompts across one or more OpenAI-compatible API clients."""

    deepseek_model_names = ["deepseek-chat", "deepseek-reasoner"]
    return_reasoning = model_name == "deepseek-reasoner"

    if not api_list:
        raise ValueError("api_list must contain at least one API key.")
    if len(base_url_list) == 1 and len(api_list) > 1:
        base_url_list = base_url_list * len(api_list)
    if len(base_url_list) != len(api_list):
        raise ValueError("base_url_list must contain one URL or match api_list length.")

    client_list: list[AsyncOpenAI] = [
        AsyncOpenAI(
            api_key=api,
            base_url=base_url,
            timeout=600
        )
    for api, base_url in zip(api_list, base_url_list)
    ]

    semaphore_list = [asyncio.Semaphore(api_call_limit) for _ in api_list]
    client_semaphore_cycle = cycle(zip(client_list, semaphore_list))

    tasks = []
    for i, user_prompt in enumerate(user_prompt_list):
        client, semaphore = next(client_semaphore_cycle)
        task = asyncio.create_task(call_with_index(
                                                    index=i,
                                                    client=client,
                                                    model_name=model_name,
                                                    prompt=user_prompt,
                                                    return_reasoning=return_reasoning,
                                                    max_retry=max_retry,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    max_completion_tokens=max_completion_tokens,
                                                    semaphore=semaphore,
                                                    ))
        tasks.append(task)

    results = [None] * len(user_prompt_list)
    # for finished in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"{model_name} API calls"):
    for finished in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc=f"{model_name} API calls"):
        index, result = await finished
        results[index] = result

    return results


def api_call(
        model_name: str, 
        user_prompt_list, 
        api_list: list[str], 
        base_url_list: list[str],
        api_call_limit: int,
        max_retry: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_completion_tokens: int = 512):
    """Synchronous wrapper around the async batch API caller."""
    
    return asyncio.run(api_call_async(model_name=model_name,  
                                      user_prompt_list=user_prompt_list, 
                                      api_list=api_list, 
                                      base_url_list=base_url_list,
                                      api_call_limit=api_call_limit, 
                                      max_retry=max_retry, 
                                      temperature=temperature, 
                                      top_p=top_p, 
                                      max_completion_tokens=max_completion_tokens))
