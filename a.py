import cv2
import os
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Frame, StringVar, ttk
from PIL import Image, ImageTk

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    return file_path

def compute_histogram(image):
    hist = []
    for i in range(3):  # for B, G, R channels
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist.append(channel_hist)
    return hist

def compare_histograms(query_hist, target_hist):
    score = 0
    for i in range(3):  # for B, G, R channels
        score += cv2.compareHist(query_hist[i], target_hist[i], cv2.HISTCMP_CHISQR)
    return score

def create_histogram_image(image):
    color = ('b', 'g', 'r')
    histogram_image = np.zeros((300, 256, 3), dtype=np.uint8)
    
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(hist)
        for h in range(256):
            cv2.line(histogram_image, (h, 0), (h, hist[h]), (int(col == 'b') * 255, int(col == 'g') * 255, int(col == 'r') * 255))
    
    histogram_image = cv2.flip(histogram_image, 0)
    return histogram_image

def process_and_display():
    query_image_path = choose_file()
    query_image = cv2.imread(query_image_path)
    query_hist = compute_histogram(query_image)

    folder_path = os.path.dirname(query_image_path)
    scores = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            target_image_path = os.path.join(folder_path, filename)
            target_image = cv2.imread(target_image_path)
            target_hist = compute_histogram(target_image)
            scores[target_image_path] = compare_histograms(query_hist, target_hist)

    # Sort the scores to get top 3 matches
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    top_3_matches = [item[0] for item in sorted_scores[:3]]

    # Display results in Tkinter
    display_in_tkinter(query_image_path, top_3_matches)

def display_in_tkinter(query_image_path, top_3_matches):
    # Clear the frame
    for widget in result_frame.winfo_children():
        widget.destroy()

    # Display Query title
    query_title = ttk.Label(result_frame, text="Query Image", font=("Arial", 12, "bold"))
    query_title.grid(row=0, column=0, padx=10, pady=10)

    # Load and display the query image
    query_img = Image.open(query_image_path)
    query_img = query_img.resize((200, 200))  # Resizing for display
    query_photo = ImageTk.PhotoImage(query_img)
    query_label = ttk.Label(result_frame, image=query_photo)
    query_label.grid(row=1, column=0, padx=10, pady=10)
    query_label.image = query_photo

    # Display histogram for query image
    query_hist_img = create_histogram_image(cv2.imread(query_image_path))
    query_hist_img = Image.fromarray(cv2.cvtColor(query_hist_img, cv2.COLOR_BGR2RGB))
    query_hist_img = query_hist_img.resize((200, 200))
    query_hist_photo = ImageTk.PhotoImage(query_hist_img)
    query_hist_label = ttk.Label(result_frame, image=query_hist_photo)
    query_hist_label.grid(row=2, column=0, padx=10, pady=10)
    query_hist_label.image = query_hist_photo
    
        # Load and display the top 3 matches
    for idx, match in enumerate(top_3_matches):
        # Display Match title
        match_title = ttk.Label(result_frame, text=f"Match {idx + 1}", font=("Arial", 12, "bold"))
        match_title.grid(row=0, column=idx+1, padx=10, pady=10)
        
        match_img = Image.open(match)
        match_img = match_img.resize((200, 200))  # Resizing for display
        match_photo = ImageTk.PhotoImage(match_img)
        match_label = ttk.Label(result_frame, image=match_photo)
        match_label.grid(row=1, column=idx+1, padx=10, pady=10)
        match_label.image = match_photo

        # Display histogram for matched image
        match_hist_img = create_histogram_image(cv2.imread(match))
        match_hist_img = Image.fromarray(cv2.cvtColor(match_hist_img, cv2.COLOR_BGR2RGB))
        match_hist_img = match_hist_img.resize((200, 200))
        match_hist_photo = ImageTk.PhotoImage(match_hist_img)
        match_hist_label = ttk.Label(result_frame, image=match_hist_photo)
        match_hist_label.grid(row=2, column=idx+1, padx=10, pady=10)
        match_hist_label.image = match_hist_photo

# Main program
root = Tk()
root.title("Image Matching App")
root.geometry("900x700")

# Button to select and process the query image
select_button = ttk.Button(root, text="Select Query Image", command=process_and_display)
select_button.pack(pady=20)

# Frame to display results
result_frame = ttk.Frame(root)
result_frame.pack(pady=20, padx=20)

root.mainloop()