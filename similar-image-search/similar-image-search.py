import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import tensorflow_datasets as tfds

train_ds, validation_ds = tfds.load(
    "tf_flowers", split=["train[:85%]", "train[85%:]"], as_supervised=True
)

IMAGE_SIZE = 224
NUM_IMAGES = 1000

images = []
labels = []

for (image, label) in train_ds.take(NUM_IMAGES):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    images.append(image.numpy())
    labels.append(label.numpy())

images = np.array(images)
labels = np.array(labels)

'''
!wget -q https://git.io/JuMq0 -O flower_model_bit_0.96875.zip
!unzip -qq flower_model_bit_0.96875.zip
'''
#load pretrained model
bit_model = tf.keras.models.load_model("flower_model_bit_0.96875")
print(bit_model.count_params())

#create embedding model for getting feature representation using the pretrained model
embedding_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(scale=1.0 / 255),
        bit_model.layers[1],
        tf.keras.layers.Normalization(mean=0, variance=1),
    ],
    name="embedding_model",
)

embedding_model.summary()

#Functions to create compressed representations of the embeddings
def hash_func(embedding, random_vectors):
    embedding = np.array(embedding)

    # Random projection.
    bools = np.dot(embedding, random_vectors) > 0
    return [bool2int(bool_vec) for bool_vec in bools]


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y

#Class table to used to create hash tables
class Table:
    def __init__(self, hash_size, dim):
        self.table = {}
        self.hash_size = hash_size
        self.random_vectors = np.random.randn(hash_size, dim).T

    def add(self, id, vectors, label):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}

        # Compute the hash values.
        hashes = hash_func(vectors, self.random_vectors)

        # Add the hash values to the current table.
        for h in hashes:
            if h in self.table:
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, vectors):
        # Compute hash value for the query vector.
        hashes = hash_func(vectors, self.random_vectors)
        results = []

        # Loop over the query hashes and determine if they exist in
        # the current table.
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results

#to create multiple tables
class LSH:
    def __init__(self, hash_size, dim, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size, dim))

    def add(self, id, vectors, label):
        for table in self.tables:
            table.add(id, vectors, label)

    def query(self, vectors):
        results = []
        for table in self.tables:
            results.extend(table.query(vectors))
        return results


class BuildLSHTable:
    def __init__(
        self,
        prediction_model,
        concrete_function=False,
        hash_size=8,
        dim=2048,
        num_tables=10, #number of tables depends on the number of classes
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.dim, self.num_tables)

        self.prediction_model = prediction_model
        self.concrete_function = concrete_function

    def train(self, training_files):
        for id, training_file in enumerate(training_files):
            # Unpack the data.
            image, label = training_file
            if len(image.shape) < 4:
                image = image[None, ...]

            # Compute embeddings and update the LSH tables.
            # More on `self.concrete_function()` later.
            if self.concrete_function:
                features = self.prediction_model(tf.constant(image))[
                    "normalization"
                ].numpy()
            else:
                features = self.prediction_model.predict(image)
            self.lsh.add(id, features, label)

    def query(self, image, verbose=True):
        # Compute the embeddings of the query image and fetch the results.
        if len(image.shape) < 4:
            image = image[None, ...]

        if self.concrete_function:
            features = self.prediction_model(tf.constant(image))[
                "normalization"
            ].numpy()
        else:
            features = self.prediction_model.predict(image)

        results = self.lsh.query(features)
        if verbose:
            print("Matches:", len(results))

        # Calculate Jaccard index to quantify the similarity.
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.dim
        return counts

def warmup():
    dummy_sample = tf.ones((1, IMAGE_SIZE, IMAGE_SIZE, 3))
    for _ in range(100):
        _ = embedding_model.predict(dummy_sample)

warmup()

training_files = zip(images, labels)
lsh_builder = BuildLSHTable(embedding_model)
lsh_builder.train(training_files)

validation_images = []
validation_labels = []

for image, label in validation_ds.take(100):
    image = tf.image.resize(image, (224, 224))
    validation_images.append(image.numpy())
    validation_labels.append(label.numpy())

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
validation_images.shape, validation_labels.shape

def plot_images(images, labels):
    plt.figure(figsize=(20, 10))
    columns = 5
    for (i, image) in enumerate(images):
        ax = plt.subplot(len(images) / columns + 1, columns, i + 1)
        if i == 0:
            ax.set_title("Query Image\n" + "Label: {}".format(labels[i]))
        else:
            ax.set_title("Similar Image # " + str(i) + "\nLabel: {}".format(labels[i]))
        plt.imshow(image.astype("int"))
        plt.axis("off")



def visualize_lsh(lsh_class):
    idx = np.random.choice(len(validation_images))
    image = validation_images[idx]
    label = validation_labels[idx]
    results = lsh_class.query(image)

    candidates = []
    labels = []
    overlaps = []

    for idx, r in enumerate(sorted(results, key=results.get, reverse=True)):
        if idx == 4:
            break
        image_id, label = r.split("_")[0], r.split("_")[1]
        candidates.append(images[int(image_id)])
        labels.append(label)
        overlaps.append(results[r])

    candidates.insert(0, image)
    labels.insert(0, label)

    plot_images(candidates, labels)

for _ in range(5):
    visualize_lsh(lsh_builder)
    plt.show()

def benchmark(lsh_class):
    warmup()

    start_time = time.time()
    for _ in range(1000):
        image = np.ones((1, 224, 224, 3)).astype("float32")
        _ = lsh_class.query(image, verbose=False)
    end_time = time.time() - start_time
    print(f"Time taken: {end_time:.3f}")


benchmark(lsh_builder)


