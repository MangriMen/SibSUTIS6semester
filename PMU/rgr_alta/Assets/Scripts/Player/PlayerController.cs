using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PlayerController : MonoBehaviour
{
    private Vector2 gravity = new Vector2(9.81f * 250, 0f);

    [SerializeField]
    private GameObject screenOfDeadPrefab;
    [SerializeField]
    private Canvas cnv;
    [SerializeField]
    private Text scoreText;

    private float timer = 0;
    private ulong score = 0;

    private GameObject screenOfDead;

    private Rigidbody2D rb;
    private SpriteRenderer sr;

    private bool isGrounded = false;

    public Transform left;
    public Transform right;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        sr = GetComponent<SpriteRenderer>();
        score = 0;
        scoreText.text = "0";
    }

    void Update()
    {
        if (isGrounded && (Input.touchCount > 0 && Input.GetTouch(0).phase == TouchPhase.Began || Input.GetButtonDown("Jump")))
        {
            gravity = -gravity;
            sr.flipX = !sr.flipX;
            isGrounded = false;
        }

        rb.AddForce(gravity * Time.deltaTime);

        timer += Time.deltaTime;

        if (timer > 0.1f)
        {
            score += 1;
            scoreText.text = score.ToString();
            timer = 0;
        }

    }

    private void FixedUpdate()
    {
        isGrounded = false;

        foreach(Collider2D col in getLeftRightColliders())
        {
            if (col.gameObject.layer == 6)
            {
                isGrounded = true;
                break;
            }
            else if (col.gameObject.CompareTag("KillBox"))
            {
                Destroy(gameObject, 0.8f);
                if (screenOfDead == null)
                {
                    screenOfDead = Instantiate(screenOfDeadPrefab, cnv.transform);
                    screenOfDead.transform.Find("Restart").GetComponent<Button>().onClick.AddListener(screenOfDead.GetComponent<PauseController>().onRestart);
                }
            }
        }
    }

    private List<Collider2D> getLeftRightColliders()
    {
        List<Collider2D> all = new List<Collider2D>();
        all.AddRange(Physics2D.OverlapCircleAll(left.transform.position, 0.01f));
        all.AddRange(Physics2D.OverlapCircleAll(right.transform.position, 0.01f));
        return all;
    }
}