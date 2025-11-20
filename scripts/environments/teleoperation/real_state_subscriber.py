import threading
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RealStateSubscriber(Node):
    def __init__(self, topic: str = 'vrpolicy_obs_publisher'):
        super().__init__('real_state_subscriber')
        self.sub = self.create_subscription(
            String, topic, self._callback, 30
        )
        self._lock = threading.Lock()
        self._latest: dict | None = None

    def _callback(self, msg: String):
        data = json.loads(msg.data)
        with self._lock:
            self._latest = data

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest