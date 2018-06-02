from collections import deque

class Vehicle():
    def __init__(self):
        # recent boxes deque 
        self.recent_boxes = deque(maxlen =5)
    

    def append_box(self, box_list):
        self.recent_boxes.append(box_list)
        total_boxes = []
        num = len(self.recent_boxes)
        for i in range(num):
            total_boxes.extend(self.recent_boxes[i])
        return total_boxes, num





























