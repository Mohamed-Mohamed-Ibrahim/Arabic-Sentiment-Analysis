import random

from locust import HttpUser, between, task

SAMPLE_TEXTS = [
    "الخدمة كانت ممتازة جدا والسائق محترم",
    "تجربة سيئة للغاية والتطبيق بطيء جدا",
    "لا بأس به ولكن يحتاج إلى بعض التحسينات",
    "الرحلة كانت مريحة ووصلت في الموعد المحدد",
    "السيارة نظيفة والكابتن كان متعاوناً جداً",
    "السعر مرتفع جداً مقارنة بالمسافة",
    "التطبيق يعلق كثيراً أثناء تحديد الموقع",
    "أفضل تطبيق استخدمته حتى الآن، شكراً لكم",
    "الرحلة مقبولة ولكن السائق تأخر قليلاً",
    "خدمة العملاء سيئة ولا ترد على الشكاوى"
]


class FastAPIUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def test_predict(self):
        payload = {"text": random.choice(SAMPLE_TEXTS)}
        self.client.post("/predict", json=payload)

    @task(1)
    def test_health(self):
        self.client.get("/health")


class BentoMLUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def test_predict_single(self):
        payload = {"requests": [{"text": random.choice(SAMPLE_TEXTS)}]}
        self.client.post("/predict", json=payload)

    @task(1)
    def test_predict_batch(self):
        payload = {
            "requests": [
                {"text": random.choice(SAMPLE_TEXTS)}
                for _ in range(random.randint(2, 5))
            ]
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def test_health(self):
        self.client.get("/readyz")