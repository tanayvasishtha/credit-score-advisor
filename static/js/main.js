document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const creditForm = document.getElementById('credit-form');
    const inputSection = document.getElementById('input-section');
    const resultsSection = document.getElementById('results-section');
    const backBtn = document.getElementById('back-btn');
    const recommendationsList = document.getElementById('recommendations-list');
    
    // Chart variables
    let scoreChart = null;
    
    // Handle form submission
    creditForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(creditForm);
        
        // Send data to server
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display results
            displayResults(data);
            
            // Hide input section, show results section
            inputSection.style.display = 'none';
            resultsSection.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
        });
    });
    
    // Handle back button click
    backBtn.addEventListener('click', function() {
        // Hide results section, show input section
        resultsSection.style.display = 'none';
        inputSection.style.display = 'block';
        
        // Reset form
        creditForm.reset();
        
        // Destroy chart to prevent duplicates
        if (scoreChart) {
            scoreChart.destroy();
        }
    });
    
    // Function to display results
    function displayResults(data) {
        // Update score details
        document.getElementById('result-current-score').textContent = data.current_score;
        document.getElementById('result-target-score').textContent = data.target_score;
        document.getElementById('result-score-gap').textContent = data.score_gap;
        document.getElementById('result-timeline').textContent = data.timeline_months;
        
        // Update recommendations
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
        
        // Update loan eligibility
        updateLoanEligibility('personal-loan-row', data.loan_eligibility.personal_loan);
        updateLoanEligibility('car-loan-row', data.loan_eligibility.car_loan);
        updateLoanEligibility('home-loan-row', data.loan_eligibility.home_loan);
        
        // Create/update score chart
        createScoreChart(data.current_score, data.target_score);
    }
    
    // Function to update loan eligibility status
    function updateLoanEligibility(rowId, isEligible) {
        const statusCell = document.querySelector(`#${rowId} .status`);
        if (isEligible) {
            statusCell.textContent = 'Eligible';
            statusCell.className = 'status eligible';
        } else {
            statusCell.textContent = 'Not Eligible';
            statusCell.className = 'status not-eligible';
        }
    }
    
    // Function to create score chart
    function createScoreChart(currentScore, targetScore) {
        const ctx = document.getElementById('scoreChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (scoreChart) {
            scoreChart.destroy();
        }
        
        // Create new chart
        scoreChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Current Score', 'Target Score'],
                datasets: [{
                    label: 'Credit Score',
                    data: [currentScore, targetScore],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(75, 192, 192, 0.7)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: Math.max(300, currentScore - 50),
                        max: Math.min(850, targetScore + 50)
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
});
