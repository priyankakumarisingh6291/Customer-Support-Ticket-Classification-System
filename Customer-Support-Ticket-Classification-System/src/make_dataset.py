"""Creates data/tickets.csv - a hand-written synthetic support-ticket dataset.

Run once:  python src/make_dataset.py
Descriptions are written by hand (with a few deliberately messy ones) and
combined with randomly assigned priority/status. A few duplicate rows and
missing values are added on purpose so Task 1 (cleaning) has real work to do.
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

TICKETS = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset email is not arriving in my inbox",
        "Unable to sign in, it says invalid username or password",
        "My account got locked after too many failed attempts",
        "The OTP for two factor authentication never comes",
        "I cannot log into the portal since this morning",
        "Reset password link has expired, please send a new one",
        "Login page keeps redirecting me back to the sign in screen",
        "Not able to access my account, wrong credentials error",
        "I am locked out of my profile and need help to sign in",
        "Forgot username, how do I recover it?",
        "Single sign on is failing for my company email",
        "Verification code is not received on my phone while logging in",
        "Password reset is not working",
        "Cannot authenticate, the system says my account is disabled",
        "Please unlock my user id, I am unable to log in",
        "Sign in button does nothing after entering my password",
        "I changed my password yesterday but the new one is rejected at login",
        "Getting 'session expired' every time I try to log in",
        "I can't login to the application",
        "Captcha on the login page keeps failing",
        "Need help recovering access to my account, forgot the password",
        "Login fails with my correct password on mobile app",
        "Security question answer not accepted when resetting password",
        "My credentials are not working after the latest update",
        "Unable to log in using Google sign in option",
        "Account locked, please reset my password urgently",
        "The login screen says user does not exist but I registered last week",
        "I keep getting logged out immediately after signing in",
        "Cannot access dashboard because I am not able to sign in",
        "Two step verification code is invalid every time",
        "Password change link opens a blank page",
    ],
    "Application Error": [
        "The application is showing an error while saving data",
        "I get a 500 internal server error when I submit the form",
        "App crashes every time I click on the save button",
        "Error message 'something went wrong' appears on checkout",
        "The page shows a blank white screen with an exception",
        "Unexpected error occurred while uploading a file",
        "Application throws a null pointer error on the orders page",
        "I got error code 404 when opening my invoice",
        "The app closed suddenly and my data was not saved",
        "Getting a database connection error on every request",
        "Submitting the form fails with a validation error even though data is correct",
        "System shows 'unable to process request' message",
        "The mobile app crashes on startup after the update",
        "Error while saving data, application gives error",
        "Download button throws an error and nothing happens",
        "Application freezes and then displays an unhandled exception",
        "Bug in the search feature, it returns an error instead of results",
        "Error 503 service unavailable when I try to open the application",
        "The export function crashes the application",
        "I see a pop up saying fatal error and the app closes",
        "Cannot create a new record, system returns an error",
        "Payment page gives an unknown error and the transaction fails",
        "The application stopped working after I clicked submit",
        "Getting timeout error message when updating my project",
        "A script error appears on the home page every time it loads",
        "Application error: failed to load module",
        "Data is not getting saved and an error appears on the screen",
        "Attachment upload fails with an error message",
        "Page crashes when I open the settings menu",
        "There is a bug, the edit screen shows an error and closes",
        "The application displays error code 1024 randomly",
        "Delete option gives an exception and the record stays",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "Need the quarterly revenue report for the finance team",
        "I want to download the attendance report for last month",
        "Can you share the weekly performance summary report?",
        "The sales report shows wrong totals, please check",
        "How do I export the report to Excel?",
        "Need a custom report of all closed tickets this year",
        "Request for annual expense report in PDF format",
        "The report is not generating for the selected date range",
        "Please schedule a daily inventory report to my email",
        "I need a report of customer orders by region",
        "Monthly sales report",
        "Add a column for discount in the sales report",
        "Can I get a summary report of last quarter's invoices?",
        "The downloaded report is missing some rows",
        "Need a dashboard report showing team productivity",
        "Report data does not match what I see on the screen",
        "Please generate the profit and loss statement for October",
        "Looking for the audit report of user activities",
        "Requesting a detailed report on product returns",
        "Want to see a report comparing this year with last year",
        "Report template needs to include our company logo",
        "Unable to find where to run the monthly billing report",
        "Please send me the stock levels report for all warehouses",
        "I need a report on employee leave balances",
        "Could you prepare the customer feedback report for November?",
        "The scheduled report was not delivered this week",
        "Need analytics report of website visitors for the last 30 days",
        "Please provide report of pending payments",
        "Generate sales report by product category",
        "Request a copy of the compliance report",
        "Report export to CSV has wrong date format",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on the account",
        "I want to change the name on my profile",
        "Need to update my billing address",
        "Please change my registered phone number to a new one",
        "How can I update my company details in my account?",
        "My date of birth is wrong in the profile, please correct it",
        "I would like to add a secondary email to my account",
        "Update my shipping address for future orders",
        "Please remove my old phone number from the account",
        "I want to change my account username",
        "Need to update the nominee details on my profile",
        "Change registered email id from the old one to the new one",
        "Kindly update the spelling of my name on my account",
        "I moved to a new city, please update my address",
        "Want to change the notification preferences on my account",
        "Need to upgrade my account plan to premium",
        "Please update my GST number in my account details",
        "I would like to close my account and delete my data",
        "Update my contact details, my number has changed",
        "Please link my new bank account to my profile",
        "How do I change the language and time zone on my account?",
        "Replace the profile picture on my account",
        "Please transfer the account ownership to my colleague",
        "I need to edit my mailing address in the profile",
        "My registered mobile number is no longer active, please change it",
        "Request to update the department name on my account",
        "Please correct the company name on my account",
        "Add another user to my account and update the permissions",
        "I want to update my PAN details in my profile",
        "Change my email address, the old one is not in use anymore",
        "Please modify the contact person for our account",
    ],
    "Performance": [
        "The application is very slow today",
        "Pages take too long to load since the morning",
        "The app is lagging badly when I scroll through the list",
        "Dashboard takes more than a minute to open",
        "System is extremely slow and keeps hanging",
        "The website is taking too long to respond",
        "Search results are very slow, it takes ages",
        "The application is taking too long to load",
        "Uploading files is painfully slow",
        "Everything is slow after the last update",
        "Screen freezes for a few seconds on every click",
        "Slow response time while opening reports page",
        "The mobile app is sluggish and takes time to start",
        "Application performance has degraded a lot this week",
        "It takes forever to save a record",
        "Loading time of the home page is very high",
        "The system becomes slow during peak hours",
        "Very high latency when I use the application from the office network",
        "The app is slow and unresponsive",
        "Opening any module takes a really long time",
        "Page keeps loading and loading, performance is terrible",
        "Typing in the form fields is delayed and laggy",
        "Our team is facing slowness in the system since Monday",
        "The software is working very slowly on my laptop",
        "Application speed is too low compared to last month",
        "Video streaming inside the app buffers continuously",
        "Why is the website so slow today?",
        "Data sync is taking hours instead of minutes",
        "The app hangs for a long time when I open large files",
        "Response is slow while filtering the data table",
        "System lag is affecting our daily work",
        "The portal is crawling, please look into the speed issue",
    ],
    "Billing": [
        "I was charged twice for the same subscription",
        "Please send me a copy of my last invoice",
        "I want a refund for the payment I made yesterday",
        "My credit card was charged but the order is not confirmed",
        "The invoice amount is higher than what was agreed",
        "Need to cancel my subscription and stop further charges",
        "Payment failed but the amount was deducted from my bank",
        "Why was I billed for a plan I did not choose?",
        "Please apply the discount coupon to my last bill",
        "Request for a GST invoice for last month's payment",
        "Refund has not been received yet after 10 days",
        "There is an extra charge on my monthly bill",
        "How can I change my payment method to a different card?",
        "I did not receive a receipt after making the payment",
        "Overcharged this month, please check my billing",
        "Please explain the tax amount shown on my invoice",
        "Auto renewal charged me even though I cancelled",
        "I need a duplicate bill for my records",
        "Money was debited twice for one transaction",
        "Requesting a credit note for the wrong invoice",
        "My payment is still showing as pending",
        "Need the billing statement for the last six months",
        "Please correct the company name on the invoice",
        "Unexpected deduction from my account for the premium plan",
        "I want to know why my bill increased this month",
        "Cancel my order and refund the money to my card",
        "Subscription fee was taken after the trial ended without notice",
        "Please resend the payment confirmation email",
        "Discount promised by the sales team is missing in the invoice",
        "Invoice shows wrong billing address",
        "Need to pay my pending dues, send me the payment link",
        "Charged in dollars instead of rupees, please fix the billing",
    ],
}


PRIORITY_W = {
    "Login Issue":       (0.55, 0.35, 0.10),
    "Application Error": (0.60, 0.30, 0.10),
    "Report":            (0.10, 0.55, 0.35),
    "Account Update":    (0.10, 0.50, 0.40),
    "Performance":       (0.45, 0.40, 0.15),
    "Billing":           (0.35, 0.45, 0.20),
}
STATUS_W = (0.40, 0.25, 0.35) 


def main():
    rows = []
    for cat, descs in TICKETS.items():
        for d in descs:
            pr = random.choices(["High", "Medium", "Low"], PRIORITY_W[cat])[0]
            st = random.choices(["Open", "In Progress", "Closed"], STATUS_W)[0]
            rows.append({"ticket_description": d, "category": cat, "priority": pr, "status": st})
    random.shuffle(rows)

    
    for r in random.sample(rows, 6):
        dup = dict(r)
        dup["ticket_description"] = random.choice([r["ticket_description"].upper(),
                                                   "  " + r["ticket_description"] + "  ",
                                                   r["ticket_description"]])
        rows.append(dup)
    random.shuffle(rows)

    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"T{i:03d}" for i in range(1, len(df) + 1)])
    df.insert(2, "customer_name", [random.choice(["Aarav", "Diya", "Rohan", "Sneha", "Arjun", "Meera",
                                                 "Kabir", "Ananya", "Vikram", "Ishita"]) for _ in range(len(df))])
    df.insert(3, "date", pd.to_datetime("2026-09-01") + pd.to_timedelta([random.randint(0, 60) for _ in range(len(df))], unit="D"))
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Missing values on purpose (priority/status only; description/category are needed)
    for col, n in (("priority", 4), ("status", 3), ("customer_name", 3)):
        df.loc[random.sample(range(len(df)), n), col] = None
    # one empty description (cannot be used for training)
    df.loc[random.randrange(len(df)), "ticket_description"] = None

    out = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
