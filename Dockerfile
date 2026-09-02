ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Optional production planner backend. The node still defaults to bootstrap,
# but `planner_backend=langchain_openai` must work inside the image when a key
# and model are configured.
RUN pip install --no-cache-dir --break-system-packages \
    langchain-openai \
    langchain-core

COPY package.xml setup.py setup.cfg /ros2_ws/src/mc_ai_bt/
COPY resource /ros2_ws/src/mc_ai_bt/resource
COPY launch /ros2_ws/src/mc_ai_bt/launch
COPY mc_ai_bt /ros2_ws/src/mc_ai_bt/mc_ai_bt

WORKDIR /ros2_ws
RUN . /opt/ros/jazzy/setup.sh && \
    . /ros2_ws/install/setup.sh && \
    colcon build --packages-select mc_ai_bt
