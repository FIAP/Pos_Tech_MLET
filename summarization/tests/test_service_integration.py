import bentoml
import subprocess

from service import EXAMPLE_INPUT

def test_summarization_service_integration():
    with subprocess.Popen(["bentoml", "serve", "service:Summarization", "-p", "50001"]) as server_proc:
        try:
            client = bentoml.SyncHTTPClient("http://localhost:50001", server_ready_timeout=30)
            summarized_text = client.summarize(texts=EXAMPLE_INPUT)

            # Ensure the summarized text is not empty
            assert summarized_text, "The summarized text should not be empty."
            # Check the type of the response
            assert isinstance(summarized_text, list), "The response should be a list."
            # Verify the length of the summarized text is less than the original input
            print(len(summarized_text))
            print(len(EXAMPLE_INPUT))
            assert len(summarized_text[0]) < len(EXAMPLE_INPUT[0]), "The summarized text should be shorter than the input."
        finally:
            server_proc.terminate()