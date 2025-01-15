from locust import HttpUser
from locust import between
from locust import task

from service import EXAMPLE_INPUT


class SummarizerTestUser(HttpUser):
    @task
    def summarize(self):
        url = "/summarize"
        self.client.post(url, json={"texts": EXAMPLE_INPUT})

    wait_time = between(0.05, 2)
