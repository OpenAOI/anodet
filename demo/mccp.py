from multicamcomposepro.augment import DataAugmenter
from multicamcomposepro.camera import CameraManager
from multicamcomposepro.utils import (CameraConfigurator, CameraIdentifier,
                                      Warehouse)


def main():
    # Setting necessary variables:
    object_var = "dwn-001"
    anomaly_var = ["Case"]
    num_train_images = 5
    num_test_images = 2
    temperature_var = 0.1
    num_aug_images = 3

    # Create a Camera Identifier instance to create camera_config.json and set camera order
    c_i = CameraIdentifier()
    c_i.camera_identifier()

    # Create a Camera Configurator instance to set camera parameters, like White Balance, Exposure, etc.

    c_c = CameraConfigurator()
    c_c.camera_configurator()

    # Create a Warehouse instance to store the data
    warehouse = Warehouse()
    warehouse.build(object_name=object_var, anomalies=anomaly_var)
    print(warehouse)

    # Create Camera Manager instance to capture images from the cameras

    camera_manager = CameraManager(
        warehouse=warehouse,
        train_images=num_train_images,
        test_anomaly_images=num_test_images,
    )
    camera_manager.run()

    # Create Data Augmentor instance to augment the images

    augmentor = DataAugmenter(
        object_name=object_var,
        temperature=temperature_var,
        num_augmented_images=num_aug_images,
    )
    if num_train_images <= 10:
        augment = input("Do you wish to create synthetic train images by augmenting existing? [Y/N] ")
        if augment.lower() != "y" and augment.lower() != "yes":
            print("Augment cancelled.")
        else:
            augmentor.augment_images()




if __name__ == "__main__":
    main()


# You can also run main like this:
# from multicamcomposepro.main import main

# if __name__ == "__main__":
#     main()
