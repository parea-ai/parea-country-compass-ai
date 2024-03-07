from langchain.tools import tool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from parea import trace, trace_insert
from parea.schemas import TraceLogImage


@tool
@trace
def countries_image_generator(country: str) -> str:
    """Call this to get an image of a country"""
    res = DallEAPIWrapper(model="dall-e-3").run(
        f"You generate image of a country representing the most "
        f"typical country's characteristics, incorporating its flag. "
        f"the country is {country}"
    )

    answer_to_agent = (
        f"Use this format- Here is an image of {country}: [{country} Image] url={res}"
    )
    trace_insert({"images": [TraceLogImage(url=res, caption=country)]})
    return answer_to_agent
