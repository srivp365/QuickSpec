<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/srivp365/QuickSpec">
    <img src="Assets/logo.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">QuickSpec</h3>

  <p align="center">
    <a href="https://github.com/srivp365/QuickSpec"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/srivp365/QuickSpec">View Demo</a>
    &middot;
    <a href="https://github.com/srivp365/QuickSpec/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/srivp365/QuickSpec/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/srivp365/QuickSpec)

 A TUI based RAG powered search tool for beginners to explore and ask questions about multiple uploaded datasheets without consuming large amounts of tokens.
   
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* [![uv][uv]][uv-url]
* [![Typer][Typer]][Typer-url]
* [![usearch][usearch]][usearch-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- FEATURES -->
## Features

* pymupdf4llm used to process pdfs into page objects
* RecursiveCharacterTextSplitter used to split into table aware chunks
* quantized ONNX bge-small used to embed chunks and send them into a userach index and add them in sqlite db
* rank_bm25 library used to store chunks in bm25 index 


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

You can run/try the program on this [google colab](https://colab.research.google.com/drive/1wYFFbEt_C8AER1pZf5ilAdj-n97vrzp-?usp=sharing), without installing anything locally! 

The steps below are for local usage/development.

### Prerequisites
Have [uv](https://docs.astral.sh/uv/) installed!

* Macos/Linux
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* Windows
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* pip
  ```sh
  pip install uv
  ```


### Installation

1. Get a free API Key at [OpenRouter](https://openrouter.ai/)
2. Run the uv tool intall command
   ```sh
   uv tool install git+https://github.com/srivp365/QuickSpec.git
   ```
3. Run the following commands to download optimized onnx model and enter OpenRouter key
   ```sh
   quickspec index "/path/to/your/datasheets"
   ```
4. Proceed to Usage to see available commands!

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

```sh
quickspec ask "question"
```
**Returns answer + sources table + timing breakdown.** 
*Retrieves relevant vectors in ingested data via usearch and bm25 rank, before merging using reciprocal rank fusing (filters out outliers, including only chunk ids that exist in both indices), and runs generation via openrouter.* 

```sh
quickspec chat
```
**Same loop as the `ask` command, just in a REPL format. Loops until exit.**


```sh
quickspec stats
```
**Prints out statistics about the index. Number of chunks, Chunk types, source documents used.**


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [ ] A way for users to test their schematics against the provided docs to ensure proper wiring following document specifications (end goal of this app!)

See the [open issues](https://github.com/srivp365/QuickSpec/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

As a first-year engineering student, I built this project as a means of bettering my full stack skills, while also learning more about Python, RAG/AI Engineering and Typer.
If you see an area for optimization, have deployment advice, or just want to discuss the code, please feel free to open an Issue or start a Discussion.
Thanks again!


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/srivp365/QuickSpec/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=srivp365/QuickSpec" alt="contrib.rocks image" />
</a>




<!-- CONTACT -->
## Contact

Srivathsan Prasanna - srivprasanna@gmail.com.com

Project Link: [https://github.com/srivp365/QuickSpec](https://github.com/srivp365/QuickSpec)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/srivp365/QuickSpec.svg?style=for-the-badge
[contributors-url]: https://github.com/srivp365/QuickSpec/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/srivp365/QuickSpec.svg?style=for-the-badge
[forks-url]: https://github.com/srivp365/QuickSpec/network/members
[stars-shield]: https://img.shields.io/github/stars/srivp365/QuickSpec.svg?style=for-the-badge
[stars-url]: https://github.com/srivp365/QuickSpec/stargazers
[issues-shield]: https://img.shields.io/github/issues/srivp365/QuickSpec.svg?style=for-the-badge
[issues-url]: https://github.com/srivp365/QuickSpec/issues
[license-shield]: https://img.shields.io/github/license/srivp365/QuickSpec.svg?style=for-the-badge
[license-url]: https://github.com/srivp365/QuickSpec/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/srivp
[product-screenshot]: Assets/banner.png


<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Python]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[Python-url]: https://www.python.org/
[uv]: https://img.shields.io/badge/uv-261230.svg?logo=uv&logoColor=#de5fe9
[uv-url]: https://docs.astral.sh/uv/
[Sentence Encoders]: https://img.shields.io/badge/Sentence_Encoders-4F46E5?logo=huggingface&logoColor=fff
[Sentence-Encoders-url]: https://www.sbert.net/

[Typer]: https://img.shields.io/badge/Typer-009688?logo=python&logoColor=fff
[Typer-url]: https://typer.tiangolo.com/

[usearch]: https://img.shields.io/badge/usearch-FF6F00?logo=c%2B%2B&logoColor=fff
[usearch-url]: https://github.com/unum-cloud/usearch
