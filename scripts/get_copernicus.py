import os
import requests

# ---------------- CONFIGURATION ----------------

USERNAME = "samudrakar8@gmail.com"
PASSWORD = "Babuikar@2021"

BBOX = (
    "geography'SRID=4326;"
    "POLYGON((77.58 12.92,"
    "77.62 12.92,"
    "77.62 12.98,"
    "77.58 12.98,"
    "77.58 12.92))'"
)

clear_dir = "dataset/clear"
cloudy_dir = "dataset/cloudy"

os.makedirs(clear_dir, exist_ok=True)
os.makedirs(cloudy_dir, exist_ok=True)


# ---------------- AUTHENTICATION ----------------

def get_access_token():

    auth_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )

    data = {
        "client_id": "cdse-public",
        "username": USERNAME,
        "password": PASSWORD,
        "grant_type": "password"
    }

    try:

        response = requests.post(
            auth_url,
            data=data,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["access_token"]

        print("Authentication failed.")
        print(response.text)

        return None

    except Exception as e:

        print("Authentication error:", e)
        return None


# ---------------- CLOUD COVER EXTRACTION ----------------

def extract_cloud_cover(product):

    for attr in product.get("Attributes", []):

        if attr.get("Name") == "cloudCover":

            try:
                return float(attr["Value"])

            except:
                return 100.0

    return 100.0


# ---------------- DOWNLOAD PRODUCT ----------------

def download_product(product_id, save_path, token):

    url = (
        f"https://catalogue.dataspace.copernicus.eu/"
        f"odata/v1/Products({product_id})/$value"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=300
        )

        if response.status_code == 200:

            with open(save_path, "wb") as f:

                for chunk in response.iter_content(8192):

                    if chunk:
                        f.write(chunk)

            print("Saved:", save_path)

        else:

            print("Download failed")
            print(response.status_code)

    except Exception as e:

        print("Download error:", e)


# ---------------- MAIN ----------------

def main():

    print("Authenticating...")

    token = get_access_token()

    if token is None:
        return

    print("Authentication successful\n")

    base_url = (
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    query_filter = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/Value eq 'S2MSI1C') and "
        f"OData.CSC.Intersects(area={BBOX})"
    )

    url = (
        f"{base_url}"
        f"?$filter={query_filter}"
        "&$expand=Attributes"
        "&$top=100"
    )

    print("Fetching metadata...\n")

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=120
        )

        print("Status Code:", response.status_code)

        if response.status_code != 200:

            print(response.text)
            return

        products = response.json().get("value", [])

        print("Products found:", len(products))

        if len(products) == 0:

            print("No products found.")
            return

        # Sort by cloud cover
        products.sort(
            key=extract_cloud_cover
        )

        top_clear = products[:5]
        top_cloudy = products[-5:]

        print("\nDownloading clearest images...\n")

        for i, item in enumerate(top_clear):

            cloud_cover = extract_cloud_cover(item)

            print(
                f"Clear Image {i+1} "
                f"(Cloud Cover {cloud_cover}%)"
            )

            filename = (
                os.path.join(
                    clear_dir,
                    f"pair_{i:02d}_clear.zip"
                )
            )

            download_product(
                item["Id"],
                filename,
                token
            )

        print("\nDownloading cloudiest images...\n")

        top_cloudy.reverse()

        for i, item in enumerate(top_cloudy):

            cloud_cover = extract_cloud_cover(item)

            print(
                f"Cloudy Image {i+1} "
                f"(Cloud Cover {cloud_cover}%)"
            )

            filename = (
                os.path.join(
                    cloudy_dir,
                    f"pair_{i:02d}_cloudy.zip"
                )
            )

            download_product(
                item["Id"],
                filename,
                token
            )

        print("\nFinished.")

    except Exception as e:

        print("Error:")
        print(e)


# ---------------- RUN ----------------

if __name__ == "__main__":
    main()