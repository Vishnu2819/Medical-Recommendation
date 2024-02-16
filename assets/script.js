let slideIndex = 0;

function showSlides() {
  const slides = document.querySelectorAll('.slide');
  slideIndex++;
  if (slideIndex >= slides.length) { slideIndex = 0; }
  slides.forEach((slide) => {
    slide.style.transform = `translateX(-${slideIndex * 100}%)`;
  });
}

function plusSlides(n) {
  slideIndex += n;
  showSlides();
}

setInterval(showSlides, 3000);
