import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_grayscale_image(path):
    img = Image.open(path).convert('L')  
    return np.array(img)

def compress_image_svd(img_matrix, k):
    #compute svd
    U, S, VT = np.linalg.svd(img_matrix, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    compressed = np.dot(U_k, np.dot(S_k, VT_k))
    return compressed

def plot_results(original, compressed, k):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(compressed, cmap='gray')
    axes[1].set_title(f"Compressed (k={k})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def compression_stats(img_shape, k):
    m,n = img_shape
    uncompressed_size = m * n
    compressed_size = k * (1 + m + n)  # k singular values + U_k+VT_k
    ratio = compressed_size / uncompressed_size
    print(f"Compression ratio: {ratio:.2f} (compressed is {ratio*100:.1f}% of original)")

def main():
    image_path = "input.png"  # replace with grayscale img
    img_matrix = load_grayscale_image(image_path)
    print("Image loaded with shape:", img_matrix.shape)

    k = 50 
    compressed_img = compress_image_svd(img_matrix,k)
    compressed_img = np.clip(compressed_img, 0, 255)

    plot_results(img_matrix, compressed_img,k)
    compression_stats(img_matrix.shape,k)

    #save img
    compressed_img = Image.fromarray(compressed_img.astype('uint8'))
    output_path = f"compressed_k{k}.png"
    compressed_img.save(output_path)
    print(f"Compressed image saved as {output_path}")

if __name__ == "__main__":
    main()
