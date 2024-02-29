-- Mise à jour de la table users
CREATE TABLE IF NOT EXISTS demo.users (
    user_id INT64 NOT NULL,
    username STRING,
    email STRING,
    first_name STRING,
    last_name STRING,
    registration_date DATE,
    birth_date DATE,
    country STRING
);

-- Mise à jour de la table products (aucun changement suggéré ici, mais vous pourriez vouloir ajouter plus tard)
CREATE TABLE IF NOT EXISTS demo.products (
    product_id INT64 NOT NULL,
    product_name STRING,
    product_description STRING
);

-- Mise à jour de la table comments avec une note de produit
CREATE TABLE IF NOT EXISTS demo.comments (
    comment_id INT64 NOT NULL,
    user_id INT64,
    product_id INT64,
    comment_text STRING,
    comment_date DATE,
    product_rating INT64
);


-- Insert users data
INSERT INTO demo.users (user_id, username, email, first_name, last_name, registration_date, birth_date, country) VALUES
(1, 'Alex', 'alex@example.com', 'Alex', 'Johnson', '2021-01-15', '1990-04-01', 'USA'),
(2, 'Charlie', 'charlie@example.com', 'Charlie', 'Brown', '2021-02-20', '1985-05-12', 'Canada'),
(3, 'Jordan', 'jordan@example.com', 'Jordan', 'Davis', '2021-03-25', '1992-08-23', 'UK'),
(4, 'Taylor', 'taylor@example.com', 'Taylor', 'Smith', '2021-04-30', '1989-11-09', 'Australia'),
(5, 'Morgan', 'morgan@example.com', 'Morgan', 'Lee', '2021-05-05', '1993-02-17', 'New Zealand'),
(6, 'Casey', 'casey@example.com', 'Casey', 'Wong', '2021-06-10', '1988-07-26', 'Singapore'),
(7, 'Riley', 'riley@example.com', 'Riley', 'Martin', '2021-07-15', '1995-10-30', 'Ireland'),
(8, 'Quinn', 'quinn@example.com', 'Quinn', 'ONeil', '2021-08-20', '1996-03-15', 'USA'),
(9, 'Jamie', 'jamie@example.com', 'Jamie', 'Garcia', '2021-09-25', '1984-06-04', 'Mexico'),
(10, 'Drew', 'drew@example.com', 'Drew', 'Barrett', '2021-10-30', '1991-12-22', 'USA'),
(11, 'Avery', 'avery@example.com', 'Avery', 'Chen', '2021-11-04', '1987-01-13', 'China'),
(12, 'Reese', 'reese@example.com', 'Reese', 'Kumar', '2021-12-09', '1994-09-08', 'India'),
(13, 'Peyton', 'peyton@example.com', 'Peyton', 'Lopez', '2022-01-14', '1990-05-27', 'Spain'),
(14, 'Skyler', 'skyler@example.com', 'Skyler', 'Müller', '2022-02-18', '1993-07-19', 'Germany'),
(15, 'Dakota', 'dakota@example.com', 'Dakota', 'Suzuki', '2022-03-25', '1989-03-03', 'Japan'),
(16, 'Devon', 'devon@example.com', 'Devon', 'Kim', '2022-04-29', '1995-11-11', 'South Korea'),
(17, 'Leslie', 'leslie@example.com', 'Leslie', 'Andersson', '2022-05-04', '1986-02-14', 'Sweden'),
(18, 'Kennedy', 'kennedy@example.com', 'Kennedy', 'Silva', '2022-06-09', '1992-10-26', 'Brazil'),
(19, 'Jordan', 'jordanb@example.com', 'Jordan', 'Bishop', '2022-07-14', '1990-08-05', 'USA'),
(20, 'Alex', 'alexc@example.com', 'Alex', 'Cameron', '2022-08-19', '1988-04-16', 'UK');


-- Insert products data
INSERT INTO demo.products (product_id, product_name, product_description) VALUES
(18, 'Electric Kettle', 'Fast boiling kettle with temperature control.'),
(19, 'Smart Thermostat', 'Control your home temperature remotely.'),
(20, 'Fitness Band', 'Track your fitness activity and goals.'),
(21, 'Smart Lock', 'Keyless entry and remote door locking.'),
(22, 'Digital Camera', 'Capture moments with high-quality photos.'),
(23, 'Gaming Keyboard', 'Mechanical keyboard for professional gamers.'),
(24, 'Projector', 'HD projector for movies and presentations.'),
(25, 'Smart Scale', 'Digital scale with health metrics integration.'),
(26, 'Electric Toothbrush', 'Smart toothbrush with app integration.'),
(27, 'GPS Tracker', 'Keep track of your valuable items.'),
(28, 'Smart Light Bulb', 'Control lighting from your smartphone.'),
(29, 'Air Purifier', 'Improve your indoor air quality.'),
(30, 'Smart Watch', 'Next-gen smartwatch with new features.'),
(31, 'Action Cam', 'Capture your outdoor adventures.'),
(32, 'Noise Cancelling Earbuds', 'Immersive sound experience with no distractions.'),
(33, 'Wireless Charging Pad', 'Conveniently charge your devices wirelessly.'),
(34, 'Smart Water Bottle', 'Track your hydration with a smart bottle.'),
(35, 'Smart Glasses', 'Wearable tech for augmented reality experiences.'),
(36, 'Weather Station', 'Monitor local weather conditions in real-time.'),
(37, 'Smart Mirror', 'Interactive mirror with fitness and health tracking.');


-- Continuation de l'insertion des commentaires avec ratings
INSERT INTO demo.comments (comment_id, user_id, product_id, comment_text, comment_date, product_rating) VALUES
(19, 1, 18, 'Makes morning coffee a breeze.', '2023-02-19', 2),
(20, 2, 19, 'Easily the best upgrade to my home.', '2023-02-20', 5),
(21, 3, 20, 'Helped me stay on track with my fitness.', '2023-02-21', 4),
(22, 4, 21, 'The convenience is unmatched.', '2023-02-22', 4),
(23, 5, 22, 'Takes stunning pictures every time.', '2023-02-23', 1),
(24, 6, 23, 'A must-have for any serious gamer.', '2023-02-24', 2),
(25, 7, 24, 'Movie nights have never been better.', '2023-02-25', 4),
(26, 8, 25, 'Really helps me understand my health better.', '2023-02-26', 5),
(27, 9, 26, 'Never going back to manual brushes.', '2023-02-27', 4),
(28, 10, 27, 'Found my lost luggage in no time.', '2023-02-28', 5),
(29, 11, 28, 'Lighting my room has never been this fun.', '2023-03-01', 5),
(30, 12, 29, 'My allergies have significantly improved.', '2023-03-02', 1),
(31, 13, 30, 'The new features are incredible.', '2023-03-03', 4),
(32, 14, 31, 'Rugged and reliable for all my hikes.', '2023-03-04', 5),
(33, 15, 32, 'Music sounds better without the noise.', '2023-03-05', 4),
(34, 16, 33, 'So convenient for my nightstand.', '2023-03-06', 5),
(35, 17, 34, 'I m finally drinking enough water daily.', '2023-03-07', 3),
(36, 18, 35, 'Feels like the future is here.', '2023-03-08', 4),
(37, 19, 36, 'I can keep an eye on the weather for my garden.', '2023-03-09', 5),
(38, 20, 37, 'Morning routines are now fun and informative.', '2023-03-10', 5);

