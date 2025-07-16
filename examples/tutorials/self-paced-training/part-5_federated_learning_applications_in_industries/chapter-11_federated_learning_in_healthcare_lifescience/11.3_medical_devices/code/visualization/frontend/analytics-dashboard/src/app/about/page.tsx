/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Text } from "@kui-react/text";

// Common styles

const headingStyles = {
  margin: 20,
 color: "#76b900"
}

const paragraphStyles = {
  margin: 20
}

const linkStyles = {
  borderBottom: "1px solid #76b900"
}

const listItemStyles = {
  margin: 30
}

export default function AboutPage() {
  // Display about page
  return (
    <>
        <Text as="h1" style={headingStyles}>Holoscan Federated Analytics</Text>
        <Text as="h2" style={headingStyles}>Overview</Text>
        <Text as="p" style={paragraphStyles}>Federated Analytics is an approach to user data analytics that combines information from distributed datasets without gathering it at a central location.</Text>
        <Text as="p" style={paragraphStyles}>NVIDIA Holoscan is normally deployed on a fleet of IGX devices. A goal here is to provide Holoscan customers a feature where Holoscan customers can capture & analyze specific metrics for Holoscan applications deployed on a fleet of NVIDIA IGX devices. This is demonstrated with a sample end-to-end Holoscan federated analytics application.</Text>
        <Text as="p" style={paragraphStyles}>The Holoscan Federate Analytics is using <a style={linkStyles} href="https://developer.nvidia.com/flare">NVIDIA FLARE</a>. NVIDIA FLARE supports federated statistics where FLARE provides built-in federated statistics operators (controller and executors) that can generate global statistics based on local client side statistics.</Text>
        <Text as="h2" style={headingStyles}>NVFLARE for Federated Statistics</Text>
        <Text as="p" style={paragraphStyles}>NVIDIA FLARE (NVIDIA Federated Learning Application Runtime Environment) is a robust framework designed to facilitate federated learning and federated statistics. It allows multiple institutions to collaboratively compute statistical measures on distributed datasets without centralizing the data. This approach ensures data privacy and security while enabling the generation of aggregated statistical insights.</Text>
        <Text as="h3" style={headingStyles}>Key Components of NVFLARE involved in Federated Statistics</Text>
        <Text as="h4" style={paragraphStyles}>Server</Text>
        <Text as="p" style={paragraphStyles}></Text>
        <Text as="p" style={paragraphStyles}>The server is the central coordinating entity in the federated statistics workflow.</Text>
        <Text as="h4" style={paragraphStyles}>NVFLARE Admin Server</Text>
        <Text as="p" style={paragraphStyles}>It manages control operations, such as starting, stopping, and monitoring the federated statistics process.</Text>
        <Text as="h4" style={paragraphStyles}>NVFLARE App Server</Text>
        <Text as="p" style={paragraphStyles}>It orchestrates the aggregation of statistical results from multiple clients, ensuring that global statistical measures are computed correctly.</Text>
        <Text as="h4" style={paragraphStyles}>NVFLARE Client</Text>
        <Text as="p" style={paragraphStyles}>Clients are the participants that perform local statistical computations on their datasets.</Text>
        <Text as="p" style={paragraphStyles}>Local Statistician: The component responsible for calculating local statistical measures such as mean, variance, and standard deviation on the client’s dataset.</Text>
        <Text as="p" style={paragraphStyles}>Communicator: Handles the communication between the client and the server, including sending local statistical results to the server and receiving aggregated statistics.</Text>
        <Text as="h4" style={paragraphStyles}>Federated Statistics Workflow</Text>
        <Text as="p" style={paragraphStyles}>The workflow defines the process and sequence of tasks for federated statistics computation.</Text>
        <Text as="p" style={paragraphStyles}>Statistics Workflow: Specifies the steps involved in computing and aggregating statistics, including local computation and aggregation tasks.</Text>
        <Text as="p" style={paragraphStyles}>Task Definitions: Detailed definitions of specific tasks, such as computing local statistics and aggregating results, which are executed during the workflow.</Text>
        <Text as="h4" style={paragraphStyles}>Configuration Files</Text>
        <Text as="p" style={paragraphStyles}>Configuration files define the settings and parameters for the federated statistics application.</Text>
        <Text as="p" style={paragraphStyles}>Server Configuration: Contains settings related to the server’s role, such as aggregation methods and communication protocols.</Text>
        <Text as="p" style={paragraphStyles}>Client Configuration: Specifies parameters for the client’s local computations, including the statistical measures to compute.</Text>
        <Text as="p" style={paragraphStyles}>Workflow Configuration: Describes the sequence of tasks in the federated statistics workflow and their respective parameters.</Text>
        <Text as="h4" style={paragraphStyles}>Example Workflow in Federated Statistics</Text>
        <ul>
        <Text as="li" style={listItemStyles}>Initialization: Clients and server initialize the federated learning environment.</Text>
        <Text as="li" style={listItemStyles}>Local Computation: Each client computes local statistical measures (e.g., count, mean, variance) on its own data.</Text>
        <Text as="li" style={listItemStyles}>Communication: Clients send their local statistics to the server.</Text>
        <Text as="li" style={listItemStyles}>Aggregation: The server aggregates the local statistics into global statistics.</Text>
        <Text as="li" style={listItemStyles}>Distribution: The server sends the aggregated statistics back to the clients.</Text>
        <Text as="li" style={listItemStyles}>Evaluation: Clients may perform local evaluation of the aggregated statistics.</Text>
        </ul>
        <Text as="h4" style={paragraphStyles}>Summary</Text>
        <Text as="p" style={paragraphStyles}>NVFLARE in the context of federated statistics provides a comprehensive framework for securely and collaboratively computing statistical measures across distributed datasets. By leveraging its key components—server, client, federated statistics workflow, configuration files, data management, security and privacy, logging and monitoring, and extensibility—NVFLARE ensures a robust, secure, and flexible platform for federated statistics applications.</Text>
        <Text as="h2" style={headingStyles}>Holoscan Federated Analytics using NVFLARE</Text>
        <Text as="p" style={paragraphStyles}>NVIDIA Holoscan is the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, and Industrial Inspection.</Text>
        <Text as="p" style={paragraphStyles}>NVIDIA Holoscan is normally deployed on a fleet of IGX devices. A goal here is to provide Holoscan customers a feature where Holoscan customers can capture and analyze specific metrics for Holoscan applications deployed on a fleet of IGX devices. This should be demonstrated with a sample end-to-end Holoscan federated analytics application.</Text>
        <Text as="p" style={paragraphStyles}>There are three main tasks involved here:</Text>
        <ul>
        <Text as="li" style={listItemStyles}>Data Collection</Text>
        <Text as="li" style={listItemStyles}>Data Processing and</Text>
        <Text as="li" style={listItemStyles}>Data Analytics.</Text>
        <Text as="p" style={paragraphStyles}>Data collection is separately handled by Holoscan whereas NVFLARE is used for Data Processing and Data Analytics.</Text>
        </ul>
      </>
  );
    
}
