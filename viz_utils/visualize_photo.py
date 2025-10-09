import matplotlib.pyplot as plt
import  cv2


def visualize_photo_with_bboxes(photo_annotation, show_inline=True):
    image_path = photo_annotation['image_path']
    image = cv2.imread(image_path)
    
    # Draw bounding boxes
    for bbox_info in photo_annotation['bboxes']:
        bbox = bbox_info['bbox']  # [xmin, xmax, ymin, ymax]
        charname = bbox_info['charname']
        transliteration = bbox_info['transliteration']
        xmin, xmax, ymin, ymax = map(int, bbox.strip('[]').split(','))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"{charname} ({transliteration})",
                    (xmin, max(15, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
