let slideIndex = 0;

function showSlides() {
    const slides = document.getElementsByClassName("slide");
    if (slideIndex >= slides.length) {
        slideIndex = 0;
    }
    if (slideIndex < 0) {
        slideIndex = slides.length - 1;
    }
    for (let i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }
    slides[slideIndex].style.display = "block";
}

function moveSlide(n) {
    slideIndex += n;
    showSlides();
}

document.querySelector('.prev').addEventListener('click', function() {
  alert("yo!!");
    moveSlide(-1);
});

document.querySelector('.next').addEventListener('click', function() {
    moveSlide(1);
});

function autoSlideshow() {
    moveSlide(1);
    setTimeout(autoSlideshow, 2000); // Change slide every 2 seconds (adjust as needed)
}

autoSlideshow(); // Start automatic slideshow
