        if show_label:
            x, y = poly[0]
            cv2.putText(img, category, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, color, 2, cv2.LINE_AA)