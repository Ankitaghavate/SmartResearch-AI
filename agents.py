import json
from utils import call_llm


def planner(topic):
    res = call_llm(
        "Planner",
        f"Return 3 queries in JSON for topic: {topic}"
    )

    try:
        return json.loads(res)
    except:
        return [
            {"query": topic + " overview"},
            {"query": topic + " applications"},
            {"query": topic + " benefits"}
        ]


def critic(report):
    return call_llm("Critic", report)


def improver(report, critique):
    return call_llm("Improve", report + "\nFix:\n" + critique)


def verifier(report, context):
    return call_llm(
        "Fact-checker",
        f"Check hallucination:\nReport:{report}\nContext:{context}"
    )


def writer(summary):
    return call_llm("Writer", summary)


def summarizer(context):
    return call_llm("Summarize", context)
