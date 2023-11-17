import { API_URL, BUCKET_NAME, REGION, IDENTITY_POOL_ID} from "./constants.js";

export async function addPhoto() {

  AWS.config.update({
    region: REGION,
    credentials: new AWS.CognitoIdentityCredentials({
      IdentityPoolId: IDENTITY_POOL_ID
    })
  });

  var files = document.getElementById("photoupload").files;
  if (!files.length) {
    return alert("Please choose a file to upload first.");
  }
  var file = files[0];
  var fileName = file.name;

  var uuid = self.crypto.randomUUID();
  uuid = uuid.toString();
  var photoKey = uuid + '/' + fileName;

  var upload = new AWS.S3.ManagedUpload({
    params: {
      Bucket: BUCKET_NAME,
      Key: photoKey,
      Body: file
    }
  });
  var promise = upload.promise();

  promise.then(
    function() {
      alert("Successfully uploaded photo.");
      var download_url = get_presigned_url(uuid)
      download_url.then(
        function(data) {
          var download_link = document.getElementById("download-link");
          download_link.innerHTML = `<a href="${data.body}">Click to Download!</a>`
        })
    },
    function(err) {
      return alert("An error occured. Please refresh the page.", err.message);
    }
  );
}

async function get_presigned_url(uuid) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body : JSON.stringify({
      'UUID' : uuid
    })
  });
  const presigned_url = await response.json();
  return presigned_url
}