"""Bonus task: rule-based suggested replies for each predicted category.
No external API is used, so there are no keys or credentials in this project."""

RESPONSES = {
    "Login Issue": ('Please use the "Forgot Password" option on the login page to reset your password. '
                    "If the problem continues, please contact the support team."),
    "Application Error": ("We are sorry for the trouble. Please refresh the page and try again. If the error "
                          "continues, reply with a screenshot and the time it happened so our team can investigate."),
    "Report": ("You can generate reports from the Reports section. If you need a custom report, please share the "
               "date range and fields required and we will prepare it for you."),
    "Account Update": ("Please go to Profile > Settings to update your details. For sensitive changes (phone number, "
                       "email) we may ask you to verify your identity first."),
    "Performance": ("We are sorry the application feels slow. Please clear your browser cache and try again. "
                    "Our team is checking system performance and will update you shortly."),
    "Billing": ("Thanks for reaching out. Our billing team will review your invoice/payment and respond within "
                "2 working days. Please keep your transaction ID handy."),
}


def suggest_response(category: str) -> str:
    return RESPONSES.get(category, "Thank you for contacting us. A support agent will get back to you soon.")
