<h1 align="center">
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a><br>
Sinapsis OpenAI
<br>
</h1>

<h4 align="center">Templates for seamless integration with OpenAI's powerful APIs</h4>

<p align="center">
<a href="#installation">🐍  Installation</a> •
<a href="#features"> 🚀 Features</a> •
<a href="#example"> 📚 Usage example</a> •
<a href="#webapp"> 🌐 Webapp</a> •
<a href="#documentation">📙 Documentation</a> •
<a href="#license"> 🔍 License </a>
</p>

**Sinapsis OpenAI** provides a powerful and flexible implementation for leveraging OpenAI's APIs. It enables users to easily configure and run various AI tasks including chat completions, audio processing, and image generation/editing.

<h2 id="installation"> 🐍  Installation </h2>

Install using your package manager of choice. We encourage the use of <code>uv</code>

Example with <code>uv</code>:

```bash
  uv pip install sinapsis-openai --extra-index-url https://pypi.sinapsis.tech
```
 or with raw <code>pip</code>:
```bash
  pip install sinapsis-openai --extra-index-url https://pypi.sinapsis.tech
```

<h2 id="features">🚀 Features</h2>

<h3>Templates Supported</h3>

This module includes multiple templates tailored for different OpenAI tasks:

- **OpenAIChatWrapper**: Enables chat completions using OpenAI's chat API, supporting models like GPT-4 and GPT-3.5-turbo.
- **OpenAIAudioTranslationWrapper**: Translates audio input into text in another language using Whisper models.
- **OpenAIAudioTranscriptionWrapper**: Converts audio input into text, supporting multiple languages and long audio files.
- **OpenAIAudioCreationWrapper**: Generates audio from text using speech synthesis with multiple voice options.
- **OpenAIImageCreationWrapper**: Creates images from text prompts using DALL-E models, with configurable sizes and formats.
- **OpenAIImageEditionWrapper**: Edits images using OpenAI's image editing API, supporting inpainting and variations.

> [!TIP]
> Use CLI command ``` sinapsis info --all-template-names``` to show a list with all the available Template names installed with Sinapsis OpenAI.

<details>
<summary><strong><span style="font-size: 1.25em;">🌍 General Attributes</span></strong></summary>

All templates share the following attribute:
- **`model`**: Determines which model to use depending on the task chosen.
- **`openai_init`**: Dictionary of attributes related to the arguments of the OpenAI client method. Ensure that values are assigned correctly according to the official [OpenAI documentation](https://platform.openai.com/docs/overview).

</details>
<details>
<summary><strong><span style="font-size: 1.25em;">Specific Attributes</span></strong></summary>

There are some attributes specific to the templates used:
- `OpenAIChatWrapper` has one additional attribute:
    - **`create` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI create method. Ensure that values are assigned correctly according to the official [OpenAI  chat documentation](https://platform.openai.com/docs/api-reference/chat/create).
- `OpenAIAudioTranslationWrapper` has one additional attribute:
    - **`create` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI create method. Ensure that values are assigned correctly according to the official [OpenAI audio documentation](https://platform.openai.com/docs/api-reference/audio/createTranslation).
- `OpenAIAudioTranscriptionWrapper` has two additional attributes:
    - **`create` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI create method. Ensure that values are assigned correctly according to the official [OpenAI audio documentation](https://platform.openai.com/docs/api-reference/audio/createTranscription).
- `OpenAIAudioCreationWrapper` has two additional attributes:
    - **`output_dir` (str, optional)**: Relative path where the binary audio output file will be saved. Defaults to `openai/openai_audio.mp4`.
    - **`create` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI create method. Ensure that values are assigned correctly according to the official [OpenAI audio documentation](https://platform.openai.com/docs/api-reference/audio/createSpeech).
- `OpenAIImageCreationWrapper` has two additional attributes:
    - **`response_format` (Literal, optional)**: Specifies the format of the API response. Defaults to `url`.
    - **`generate` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI generate method. Ensure that values are assigned correctly according to the official [OpenAI image documentation](https://platform.openai.com/docs/api-reference/images/create).
- `OpenAIImageEditionWrapper` has three additional attributes:
    - **`path_to_mask` (str, optional)**: File path to a mask image.
    - **`path_to_image` (str, required)**: The file path of the original image to be edited. The image is expected to be in `png` format.
    - **`edit` (dict, optional)**: Dictionary of attributes related to the arguments of the OpenAI edit method. Ensure that values are assigned correctly according to the official [OpenAI image documentation](https://platform.openai.com/docs/api-reference/images/createEdit).

</details>

> [!TIP]
> Use CLI command ```sinapsis info --example-template-config TEMPLATE_NAME``` to produce an example Agent config for the Template specified in ***TEMPLATE_NAME***.

For example, for ***OpenAIImageCreationWrapper*** use ```sinapsis info --example-template-config OpenAIImageCreationWrapper``` to produce an example config like:

```yaml
agent:
  name: my_test_agent
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: OpenAIImageCreationWrapper
  class_name: OpenAIImageCreationWrapper
  template_input: InputTemplate
  attributes:
    model: 'dall-e-3'
    response_format: url
    openai_init:
      api_key: null
      organization: null
      project: null
      base_url: null
      websocket_base_url: null
      timeout: 2
      max_retries: 2
      default_headers: null
      default_query: null
      http_client: null
    generate:
      n: 1
      quality: 'standard'
      size: '256x256'
      style: 'vivid'
      user: 'my user'
      extra_headers: null
      extra_query: null
      extra_body: null
      timeout: 2
```


<h2 id='example'>📚 Usage example</h2>

Below is an example YAML configuration for chat completions using OpenAI's GPT-4. In this example, we define an agent named my_test_agent and configure a chat template to process a text input ("What is AI?"). The chat template uses GPT-4 with a temperature of 0.2 and limits the response to a maximum of 60 completion tokens. Environment variables are used for API authentication.

<details>
<summary ><strong><span style="font-size: 1.4em;">Config</span></strong></summary>

```yaml
agent:
  name: my_test_agent
  description: "Chat example"

templates:

- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}

- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text : What is AI?

- template_name: OpenAIChat
  class_name: OpenAIChatWrapper
  template_input: TextInput
  attributes:
    model : gpt-4
    openai_init: {}
    create:
      temperature: 0.2
      max_completion_tokens: 60
```
</details>
This configuration defines an **agent** and a sequence of **templates** to create a chat completion using OpenAI GPT-4.

> [!IMPORTANT]
>Attributes specified under the `*_init` keys (e.g., `openai_init`) correspond directly to the OpenAi client parameters, while the ones under the `create` are directly associated to the arguments expected in the create method. Ensure that values are assigned correctly according to the official [OpenAI documentation](https://platform.openai.com/docs/overview).
>
> The TextInput template correspond to [sinapsis-data-readers](https://github.com/Sinapsis-AI/sinapsis-data-tools/tree/main/packages/sinapsis_data_readers). If you want to use the example, please make sure you install the package.
>

To run the config, use the CLI:
```bash
sinapsis run name_of_config.yml
```

<h2 id="webapp">🌐 Webapp</h2>

The webapp provides a simple interface to generate images using OpenAI's DALL-E 3. Just input your text prompt, and the app will create and display the generated image.

> [!IMPORTANT]
> To run the app you first need to clone this repository:

```bash
git clone git@github.com:Sinapsis-ai/sinapsis-openai.git
cd sinapsis-openai
```
> [!NOTE]
> If you'd like to enable external app sharing in Gradio, `export GRADIO_SHARE_APP=True`

> [!IMPORTANT]
> You need to set your OpenAI API key, `export OPENAI_API_KEY="your-api-key-here"`

> [!TIP]
> To use different prompts, modify the text attribute in the TextInput template within the configuration file `config_image_creation.yaml` under src/sinapsis_openai/configs. For example:

```yaml
- template_name: TextInput
  class_name: TextInput
  template_input: InputTemplate
  attributes:
    text: "Your new prompt here"
```

<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">🐳 Docker</span></strong></summary>

**IMPORTANT** This docker image depends on the sinapsis:base image. Please refer to the official [sinapsis](https://github.com/Sinapsis-ai/sinapsis?tab=readme-ov-file#docker) instructions to Build with Docker.

1. **Build the sinapsis-openai image**:
```bash
docker compose -f docker/compose.yaml build
```

2. **Start the app container**:
```bash
docker compose -f docker/compose_apps.yaml up sinapsis-openai-gradio -d
```
3. **Check the status**:
```bash
docker logs -f sinapsis-openai-gradio
```
3. The logs will display the URL to access the webapp, e.g.:

NOTE: The url can be different, check the output of logs
```bash
Running on local URL:  http://127.0.0.1:7860
```
4. To stop the app:
```bash
docker compose -f docker/compose_apps.yaml down
```

</details>


<details>
<summary id="uv"><strong><span style="font-size: 1.4em;">💻 UV</span></strong></summary>

To run the webapp using the <code>uv</code> package manager, please:

1. **Create the virtual environment and sync the dependencies**:
```bash
uv sync --frozen
```
2. **Install the wheel**:
```bash
uv pip install sinapsis-openai[all] --extra-index-url https://pypi.sinapsis.tech
```

3. **Activate the environment**:
```bash
source .venv/bin/activate
```
4. **Run the webapp**:
```bash
python webapps/image_creation.py
```
5. **The terminal will display the URL to access the webapp, e.g.**:

NOTE: The url can be different, check the output of the terminal
```bash
Running on local URL:  http://127.0.0.1:7860
```

</details>

<h2 id="documentation">📙 Documentation</h2>

Documentation for this and other sinapsis packages is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)


<h2 id="license">🔍 License</h2>

This project is licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
