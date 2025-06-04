document.addEventListener('DOMContentLoaded', () => {
    // Get all nav links
    const navLinks = document.querySelectorAll('nav a');
    
    // Add smooth scrolling to all nav links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add active state to current section in view
    const sections = ['home', 'how-it-works', 'about'];
    
    function setActiveSection() {
        const scrollPosition = window.scrollY + 100; // Offset for header

        sections.forEach(sectionId => {
            const section = document.getElementById(sectionId);
            const link = document.querySelector(`nav a[href="#${sectionId}"]`);
            
            if (section && link) {
                const sectionTop = section.offsetTop;
                const sectionBottom = sectionTop + section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                    link.classList.add('text-primary-600', 'font-bold');
                    link.classList.remove('text-primary-700', 'font-medium');
                } else {
                    link.classList.remove('text-primary-600', 'font-bold');
                    link.classList.add('text-primary-700', 'font-medium');
                }
            }
        });
    }

    // Update active section on scroll
    window.addEventListener('scroll', setActiveSection);
    
    // Set initial active section
    setActiveSection();
}); 