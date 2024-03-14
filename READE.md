# SunFader

*paper: Providing Sustainable Unmanned Facial Detection and Recognition Service on Edge*

## Abstract
Facial recognition technique is used extensively in areas like online payments, education, and social media. Traditionally, these applications relied on powerful cloud-based systems, but advancements in edge computing have changed this, enabling fast and reliable local processing in complex and extreme environment. However, new challenges arise in availability and durability insurance to make the system running 24/7 with acceptable performance. This paper proposes a novel solution to these challenging settings. First, we use edge device for local data processing, reducing the need for cloud communication and enhancing user privacy. Second, we implement an adaptive control strategy to improve energy management in these devices. Lastly, we establish a solar-powered energy system to facilitate long-term device operation. Our approach strikes a balance between performance, quality, and durability, enabling facial recognition systems to work consistently and efficiently in complex environments.

## Project structure
- [algs](algs): this folder contains the core algorithm files for the research papers.
  - [main](algs/main.py): this file mainly consists of the implementation of our proposed "Ours" algorithm and three other algorithms mentioned in the paper.
  - [common_args](algs/common_args.py): this file contains the configuration classes for all the parameters of the system.
  - [decide_execute](algs/decide_execute.py): this file is the invocation of the actual face detection algorithm. It is a simple implementation, and the code can be written according to specific needs.
- [config](config): configuration information.
  - [charge_data](config/charge_data): different month's solar intensity information.
  - [common_config](config/common_config.json): a JSON file used for storing system configuration.
  - [face_recognition_model](config/face_recognition_model.yml): a face recognition model. Due to privacy concerns, users are required to input their own face information and train the model themselves.
  - [haarcascade_frontalface_alt2](config/haarcascade_frontalface_alt2.xml): a cascaded classifier file used for face detection.
- [photos](photos): store all photo information.
  - [counted](photos/counted): store processed facial information.
  - [origin](photos/origin): store raw photo information.
- [result](result): store result files.

## How to run the project
`python3 algs/main.py Ours`

