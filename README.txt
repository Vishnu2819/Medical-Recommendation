<!-- PROJECT SHIELDS -->
<a name="readme-top"></a>
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/abrarfaizmohammed)
[![ShareHub](https://img.shields.io/badge/ShareHub-green?style=for-the-badge&logoColor=white)](https://github.com/AbrarFaizMohammed/ShareHub)



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dailyprogress-9pw4.onrender.com/" target="_blank">
    <img src="https://github.com/AbrarFaizMohammed/DailyProgress/assets/131560669/a663ec66-ea7a-4081-8ee2-d8cab4ff5813" width="80" height="80">
  </a>

  <h1 align="center">SHAREHUB</h1>
</div>



<!-- ABOUT THE PROJECT -->
### About The Project
"Sharehub" emerges as an innovative e-commerce platform developed collaboratively by a dedicated team of three individuals. This dynamic web application operates on a donation-based model, inviting users to contribute items for giveaway, which are prominently featured within its catalog. Additionally, users have the freedom to request specific items they desire, fostering a sense of community and engagement.

Functioning beyond a typical exchange platform, Sharehub serves as a digital space that encourages acts of kindness and resource sharing. Through its intuitive interface and interactive features, the platform aims to create a seamless experience for users seeking to contribute or find items of interest.

The project's development journey sharpened skills in full-stack web development, database management, and user interaction design. Sharehub embodies a vision of an inclusive and compassionate online community, where individuals come together to support one another.

With a focus on user experience and community building, Sharehub provides a modern and efficient tool for facilitating connections and promoting goodwill. The platform's ethos revolves around creating a welcoming space for sharing resources and fostering a culture of generosity.

https://github.com/AbrarFaizMohammed/ShareHub/assets/131560669/b5d6df57-3682-4456-a299-c8322f15ffd9

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Node.js](https://img.shields.io/badge/Node.js-43853D?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org/)
* [![Express.js](https://img.shields.io/badge/Express.js-000000?style=for-the-badge&logo=express&logoColor=white)](https://expressjs.com/)
* [![REST API](https://img.shields.io/badge/REST%20API-007396?style=for-the-badge&logo=rest&logoColor=white)](https://en.wikipedia.org/wiki/Representational_state_transfer)
* [![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
* [![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
* [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
* [![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
* [![Mongoose](https://img.shields.io/badge/Mongoose-880000?style=for-the-badge&logo=mongoose&logoColor=white)](https://mongoosejs.com/)
* [![EJS](https://img.shields.io/badge/EJS-2B2B2B?style=for-the-badge&logo=ejs&logoColor=white)](https://ejs.co/)
* [![Nodemon](https://img.shields.io/badge/Nodemon-76D04B?style=for-the-badge&logo=nodemon&logoColor=white)](https://nodemon.io/)
* [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)








<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To start your journey with ShareHub, simply.
To get a local copy up and running follow these simple example steps.

### Prerequisites

To get started with Docker, first install Docker Desktop from the official website: <a href="https://www.docker.com/products/docker-desktop/">Docker Desktop Download</a>.
After installing Docker Desktop, you can verify that Docker is installed by opening a terminal or command prompt and typing:
```sh
docker --version
```

Login to your Docker account or create a Docker account if you do not have one, then open a terminal or command prompt and type:

```sh
docker login
```
### Installation

Congratulations on making it this far! You're now ready to dive into the ShareHub code and start exploring.<br/>

Happy coding!

1. Now you can pull the following Docker image in your terminal or command prompt using the following command:
   ```sh
   docker pull abrarfaiz96/sharehub
   ```
2. After successfully pulling the `abrarfaiz96/sharehub` Docker image, you can run the container using the following command:
   ```sh
   docker run -d -p 8000:3000 abrarfaiz96/sharehub:latest
   ```
   The command `docker run -d -p 8000:3000 abrarfaiz96/sharehub:latest` is used to run a Docker container based on the `abrarfaiz96/sharehub:latest` image. Let's break down the command:

- `docker run`: This part of the command instructs Docker to run a container.
- `-d`: This flag tells Docker to run the container in detached mode, meaning it runs the container in the background and prints the container ID.
- `-p 8000:3000`: This option specifies port mapping, where `8000` is the host port and `3000` is the container port. This means that connections made to port `8000` on the host will be forwarded to port `3000` inside the container. So, if your application inside the container is listening on port `3000`, you can access it using port `8000` on your host machine.
- `abrarfaiz96/sharehub:latest`: This is the name of the Docker image and its tag. It tells Docker which image to use for creating the container. In this case, it's using the image named `abrarfaiz96/sharehub` with the tag `latest`.

Overall, the command runs a container in detached mode, maps port `8000` on the host to port `3000` in the container, and uses the `abrarfaiz96/sharehub:latest` image.


**Note:** You can replace port number 8000 with any other port because 8000 is the host port, meaning Docker will listen for incoming connections on port 8000 of your host machine.

3. Congratulations! 🎉 You've made it this far! Now, you just need to paste the URL.
   ```sh
     http://localhost:8000/
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


