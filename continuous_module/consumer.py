import json
import sys
from datetime import datetime

from kafka import KafkaConsumer

max_message_size = 104857600  # bytes

if __name__ == '__main__':
    consumer = KafkaConsumer(
        "models",
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id=None,
        max_partition_fetch_bytes=max_message_size
    )
    for message in consumer:
        print(f"New message {datetime.now()}\n\n")
        print(json.loads(message.value))
