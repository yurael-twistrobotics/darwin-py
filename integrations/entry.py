from gust import Gust
from integration import Integration

if __name__ == "__main__":
    gust = Gust(strategy=Integration())
    gust.start()
