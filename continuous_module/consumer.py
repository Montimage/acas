import json
import sys

from kafka import KafkaConsumer

max_message_size = 104857600 #bytes

if __name__ == '__main__':
    # topic = sys.argv[1]
    # Kafka Consumer
    consumer = KafkaConsumer(
        "messages",
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id = None,
        max_partition_fetch_bytes =max_message_size
    )
    for message in consumer:
        print("New message \n\n")
        print(json.loads(message.value))