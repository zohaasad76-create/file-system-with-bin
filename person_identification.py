import numpy as np
import os
from PIL import Image
import glob

class PersonPCA:
   
    def __init__(self, variance_threshold=0.99):
        self.variance_threshold = variance_threshold
        self.mean = None
        self.std = None
        self.pca_basis = None
        self.eigenvalues = None
        
    def fit(self, images):
        n_pixels, n_images = images.shape
        self.mean = np.mean(images, axis=1, keepdims=True)
        self.std = np.std(images, axis=1, keepdims=True, ddof=1) + 1e-8
        standardized = (images - self.mean) / self.std
        U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
        eigenvalues = (S ** 2) / (n_images - 1)
        eigenvectors = U
        total_variance = np.sum(eigenvalues)
        cumsum_variance = np.cumsum(eigenvalues)
        n_components = np.searchsorted(cumsum_variance, self.variance_threshold * total_variance) + 1
        n_components = min(n_components, len(eigenvalues))
        
        self.eigenvalues = eigenvalues[:n_components]
        self.pca_basis = eigenvectors[:, :n_components]
        
        variance_explained = np.sum(self.eigenvalues) / total_variance * 100
        print(f"  Retained {n_components}/{len(eigenvalues)} components - {variance_explained:.2f}% variance")
        
    def reconstruct(self, test_image):
        standardized = (test_image - self.mean) / self.std
        coefficients = self.pca_basis.T @ standardized
        reconstructed_std = self.pca_basis @ coefficients
        reconstructed = reconstructed_std * self.std + self.mean
        loss = np.linalg.norm(test_image - reconstructed)
        
        return reconstructed, loss


def load_image_as_vector(image_path):
    
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return img_array.flatten().reshape(-1, 1)


def load_yale_faces(data_dir):
    person_images = {}
    extensions = ['*.gif', '*.centerlight', '*.glasses', '*.happy', '*.leftlight', 
                  '*.noglasses', '*.normal', '*.rightlight', '*.sad', '*.sleepy', 
                  '*.surprised', '*.wink']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, f"subject*{ext}")))
    all_files = glob.glob(os.path.join(data_dir, "subject*"))
    for f in all_files:
        if os.path.isfile(f) and not f.endswith('.txt'):
            if f not in image_files:
                image_files.append(f)
    for img_path in image_files:
        basename = os.path.basename(img_path)
        try:
            person_id = int(basename.split('.')[0].replace('subject', ''))
            if person_id not in person_images:
                person_images[person_id] = []
            person_images[person_id].append(img_path)
        except ValueError:
            continue
    for person_id in person_images:
        person_images[person_id] = sorted(person_images[person_id])
    
    return person_images


def train_test_split_per_person(person_images, n_train=10):
    train_dict = {}
    test_dict = {}
    
    for person_id, images in person_images.items():
        if len(images) < n_train + 1:
            print(f"Warning: Person {person_id} has only {len(images)} images, skipping...")
            continue
        images_shuffled = np.random.permutation(images).tolist()
        train_dict[person_id] = images_shuffled[:n_train]
        test_dict[person_id] = [images_shuffled[n_train]] 
    
    return train_dict, test_dict


def main():
    DATA_DIR = "yalefaces"
    N_TRAIN = 10
    VARIANCE_THRESHOLD = 0.99
    SAVE_DIR = "saved_models_person"
    
    print("=" * 60)
    print("Task 1: PCA-Based Person Identification")
    print("=" * 60)
    
    np.random.seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\n1. Loading YALE Face Database...")
    person_images = load_yale_faces(DATA_DIR)
    print(f"   Found {len(person_images)} persons")
    for person_id, images in sorted(person_images.items()):
        print(f"   Person {person_id:2d}: {len(images)} images")
    print(f"\n2. Splitting data ({N_TRAIN} train, 1 test per person)...")
    train_dict, test_dict = train_test_split_per_person(person_images, N_TRAIN)
    print(f"   Training on {len(train_dict)} persons")
    print("\n3. Training PCA for each person (using efficient SVD)...")
    pca_models = {}
    
    for person_id in sorted(train_dict.keys()):
        print(f"\nPerson {person_id}:")
        train_images = []
        for img_path in train_dict[person_id]:
            img_vec = load_image_as_vector(img_path)
            train_images.append(img_vec)
        
        train_matrix = np.hstack(train_images)
        pca = PersonPCA(variance_threshold=VARIANCE_THRESHOLD)
        pca.fit(train_matrix)
        pca_models[person_id] = pca
    print("\n" + "=" * 60)
    print("4. Testing (Person Identification)")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    for true_person_id in sorted(test_dict.keys()):
        for test_img_path in test_dict[true_person_id]:
            test_image = load_image_as_vector(test_img_path)
            losses = {}
            for person_id, pca in pca_models.items():
                _, loss = pca.reconstruct(test_image)
                losses[person_id] = loss
            predicted_person_id = min(losses, key=losses.get)
            
            total += 1
            if predicted_person_id == true_person_id:
                correct += 1
                result = "✓"
            else:
                result = "✗"
            
            test_filename = os.path.basename(test_img_path)
            print(f"{result} True: Person {true_person_id:2d} | "
                  f"Predicted: Person {predicted_person_id:2d} | "
                  f"Loss: {losses[predicted_person_id]:.2f} | "
                  f"{test_filename}")
    accuracy = 100 * correct / total if total > 0 else 0
    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
